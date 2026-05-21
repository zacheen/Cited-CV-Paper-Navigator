"""Streamlit RAG chatbot for arXiv computer vision papers."""

import datetime
import sys
import threading
import time

# DIAGNOSTIC: timestamp every script run so terminal output shows where
# wall-clock time goes between `streamlit run` and first paint.
_T0 = time.perf_counter()
def _ts(label: str) -> None:
    print(f"[+{time.perf_counter() - _T0:6.2f}s] {label}", flush=True)


_ts("script start (after stdlib imports)")

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from src.config import GEMINI_MODEL, LLM_BACKEND, OLLAMA_MODEL, TOP_K
from src.processing.embedder import (
    get_chunk_count_fast,
    get_collection,
    get_collection_lite,
    is_embedder_loaded,
)
from src.processing.reranker import get_reranker, is_reranker_loaded
from src.rag import download_state
from src.rag.download_state import (
    enqueue_citation_download,
    extract_inline_citations,
    get_log_snapshot,
    is_busy,
    take_log_change_signal,
)
from src.rag.generator import generate_answer_stream as ollama_generate_answer_stream
from src.rag import generator_gemini, tools as rag_tools
from src.rag.generator_gemini import (
    generate_answer_stream as gemini_generate_answer_stream,
    run_pre_rag_pass,
)
from src.rag.react_agent import ReactFinish, ReactStep, run_react
from src.rag.retriever import format_context, retrieve, retrieve_recent_papers
from src.rag.tools import (
    retrieval_query_state,
    set_collection_provider as set_react_collection_provider,
    time_range_state,
)

_ts("heavy imports done (streamlit, chromadb, google.genai, ollama, project src)")


@st.cache_resource
def get_cached_lite_collection():
    """Lightweight collection (no SentenceTransformer) for metadata-only ops:
    sidebar chunk count, citation dedup checks. Cold cost ~0.4s, so cheap
    enough to live on the first-paint path."""
    return get_collection_lite()


@st.cache_resource(show_spinner=False)
def get_cached_collection():
    """Full collection with SentenceTransformer embedder. Used by retrieve()
    and recent-paper lookups. Cold cost ~17s for the first call (loads the
    embedder from disk), then cached for the rest of the Streamlit process
    lifetime.

    ``show_spinner`` is disabled because this function is normally called
    from the background warmup thread at module load — surfacing a spinner
    there would render at the top of the page (where the warmup kickoff
    lives in script flow) and overlap the title. Streamlit's built-in
    top-right "Running..." indicator still fires if the user submits a
    prompt while warmup is still in flight."""
    return get_collection()


# Inject the cached lite collection into download_state so its background
# citation-dedup checks reuse the same handle without download_state having
# to import Streamlit. Only the lite collection is needed there because
# _is_in_db only does metadata lookups.
download_state.set_lite_collection_provider(get_cached_lite_collection)

# Same pattern for the ReAct tools: they call retrieve() under the hood and
# need the SentenceTransformer-backed collection. Wiring it through the
# Streamlit cache means a CLI test of the tools still works (falls back to
# get_collection() inside tools.py) but the UI reuses the warm cache.
set_react_collection_provider(get_cached_collection)


@st.cache_resource(show_spinner=False)
def get_cached_chunk_count() -> int:
    """Memoize the indexed-chunk count for this Streamlit process.

    The underlying ``get_chunk_count_fast()`` runs ``SELECT MAX(rowid)``
    on ChromaDB's SQLite — sub-second even on cold OS cache, but we still
    cache it so reruns don't re-issue the query. Pre-warmed by
    :func:`_kickoff_count_warmup` so even the first sidebar render hits a
    warm cache.
    """
    return get_chunk_count_fast()


@st.cache_resource
def _kickoff_count_warmup() -> bool:
    """Pre-fill the chunk-count cache in a daemon thread before the
    sidebar renders, so the count is off the first-paint critical path.

    With ``MAX(rowid)`` this is sub-second so the speedup is small in
    absolute terms — but moving it off the main thread keeps the sidebar
    paint independent of any future SQL/disk regressions.
    """
    def _warmup() -> None:
        try:
            get_cached_chunk_count()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[chunk-count-warmup] failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    thread = threading.Thread(target=_warmup, daemon=True, name="chunk-count-warmup")
    add_script_run_ctx(thread)
    thread.start()
    return True


_kickoff_count_warmup()


@st.cache_resource
def _kickoff_embedder_warmup() -> bool:
    """Pre-warm the heavy embedder cache in a daemon thread on first run.

    The user is going to retrieve() eventually, which needs the
    SentenceTransformer model (~17s cold load). Loading it in a background
    thread while the user reads the page / types their first prompt usually
    means by the time they submit, ``get_cached_collection()`` is already
    cache-warm and the answer streams immediately.

    Streamlit's ``@st.cache_resource`` is thread-safe: if the user submits
    before the warmup finishes, their main-thread ``get_cached_collection()``
    call blocks on the same internal lock and reuses the in-progress result
    — no double computation.

    Wrapped in ``@st.cache_resource`` itself so the kicker runs exactly once
    per Streamlit process (script reruns hit the cached return and skip the
    spawn). Returns a sentinel so Streamlit can cache it.
    """
    def _warmup() -> None:
        try:
            get_cached_collection()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[embedder-warmup] failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    thread = threading.Thread(target=_warmup, daemon=True, name="embedder-warmup")
    # Attach the current Streamlit ScriptRunContext to the worker so it can
    # safely interact with @st.cache_resource without the "missing
    # ScriptRunContext" warning that Streamlit otherwise logs to stderr.
    add_script_run_ctx(thread)
    thread.start()
    return True


_kickoff_embedder_warmup()


@st.cache_resource
def _kickoff_reranker_warmup() -> bool:
    """Pre-warm the cross-encoder in a daemon thread on first run.

    The cross-encoder is ~80MB and takes a few seconds to instantiate; doing
    this at module load means the first reranked query doesn't pay the cost
    on the user's critical path. Same pattern as the embedder warmup above —
    the reranker module's own lock guarantees no double-loading even if the
    user submits before this thread finishes.
    """
    def _warmup() -> None:
        try:
            get_reranker()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[reranker-warmup] failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    thread = threading.Thread(target=_warmup, daemon=True, name="reranker-warmup")
    add_script_run_ctx(thread)
    thread.start()
    return True


_kickoff_reranker_warmup()


def get_generate_answer_stream(backend: str):
    """Return the streaming generator function for the chosen backend."""
    if backend == "gemini":
        return gemini_generate_answer_stream
    return ollama_generate_answer_stream


WEEKLY_SUMMARY_PROMPT = (
    "Please summarize the newest computer vision papers from the last 7 days. "
    "Group the answer into: 1) key themes, 2) notable papers, 3) why they matter, "
    "and 4) open questions. Only use the retrieved papers and explicitly mention "
    "when the context is insufficient."
)


def _sync_time_range_to_module() -> None:
    """Push session_state time range into the module-level state used by tools."""
    tr = st.session_state.get("time_range") or {}
    time_range_state.start_date = tr.get("start_date")
    time_range_state.end_date = tr.get("end_date")


def _sync_time_range_from_module() -> None:
    """Pull module-level state back into session_state after the pre-RAG pass."""
    st.session_state["time_range"] = time_range_state.to_dict()


def _wait_for_downloads(max_seconds: float = 60.0) -> None:
    """Block until all background ingestion jobs finish, with a spinner."""
    if not is_busy():
        return
    deadline = time.monotonic() + max_seconds
    with st.spinner("Waiting for background paper downloads to finish..."):
        while is_busy() and time.monotonic() < deadline:
            time.sleep(0.3)


def _parse_iso(value: str | None) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _format_time_range_display() -> str:
    tr = st.session_state.get("time_range") or {}
    start, end = tr.get("start_date"), tr.get("end_date")
    if not start and not end:
        return "All time"
    if start and end:
        return f"{start} -> {end}"
    if start:
        return f"{start} -> ..."
    return f"... -> {end}"


def _download_attempt_footer(triggered_paths: list[str]) -> str:
    if not triggered_paths:
        return ""
    sources = " and ".join(triggered_paths)
    return (
        "\n\n---\n"
        f"_Attempted background download of cited papers ({sources}). "
        "If the answer above feels incomplete, the new papers may now be in "
        "the database — please ask again._"
    )


def run_query(
    prompt: str,
    top_k: int,
    model_name: str,
    retrieval_query: str | None = None,
    backend: str = "ollama",
    use_reranker: bool = False,
) -> None:
    """Execute one RAG query and append the result to session history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Stage 0: block if a previous download/ingest is still running.
    _wait_for_downloads()

    triggered_paths: list[str] = []

    # Reset per-turn retrieval state so a stale cleaned query from a prior
    # turn (e.g. a Gemini turn before switching to Ollama) doesn't leak in.
    # Gemini's pre-RAG pass also clears this internally; keeping the clear
    # here makes the invariant backend-agnostic.
    retrieval_query_state.clear()

    with st.chat_message("assistant"):
        with st.status("Processing your question...", expanded=True) as status:
            # Stage 1: pre-RAG function-calling pass (Gemini only).
            if backend == "gemini":
                status.update(label="Detecting query intent (date filters, citation requests)...")
                _sync_time_range_to_module()
                log_size_before = len(get_log_snapshot())
                in_flight_before = download_state._busy_count  # noqa: SLF001
                run_pre_rag_pass(prompt, model=model_name)
                _sync_time_range_from_module()
                if download_state._busy_count > in_flight_before or len(get_log_snapshot()) > log_size_before:  # noqa: SLF001
                    triggered_paths.append("path B / explicit request")
                    status.write("Tool call fired: explicit citation download requested")

            tr = st.session_state.get("time_range") or {}
            start_date = _parse_iso(tr.get("start_date"))
            end_date = _parse_iso(tr.get("end_date"))

            # Prefer the LLM's cleaned query (Stage 1 may have stripped time /
            # download noise). Fall back to caller-supplied retrieval_query,
            # then raw prompt.
            effective_retrieval_query = (
                retrieval_query
                or retrieval_query_state.cleaned
                or prompt
            )
            # Reflect the actual query sent to ChromaDB + reranker into the
            # sidebar state, regardless of whether it came from rewrite_retrieval_query
            # (Gemini), a caller override, or the raw prompt.
            retrieval_query_state.set(effective_retrieval_query)

            # Stage 2: retrieve. The label depends on whether the embedder is
            # already warm — first query in a process pays the SentenceTransformer
            # load (~17s) unless the background warmup beat them to it. The
            # reranker is much cheaper (~80MB, few seconds) and likewise warmed
            # in the background, but flag it explicitly if it's still loading.
            if not is_embedder_loaded():
                status.update(
                    label="Loading embedding model (first-time only, ~17s)..."
                )
            elif use_reranker and not is_reranker_loaded():
                status.update(label="Loading reranker (first-time only)...")
            else:
                status.update(label="Retrieving relevant papers...")
            try:
                results = retrieve(
                    effective_retrieval_query,
                    top_k=top_k,
                    start_date=start_date,
                    end_date=end_date,
                    collection=get_cached_collection(),
                    use_reranker=use_reranker,
                )
            except Exception as e:
                status.update(label=f"Retrieval failed: {e}", state="error", expanded=True)
                st.error(f"Retrieval failed: {e}. Make sure you've run the ingestion script.")
                st.stop()

            if not results:
                status.update(label="No relevant papers found", state="error", expanded=True)
                response = "I couldn't find any relevant papers. Make sure the corpus has been ingested."
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            status.write(f"Retrieved {len(results)} papers")

            # Stage 2.5: regex auto-trigger (path A) — fire and forget.
            citation_map = extract_inline_citations(results)
            for source_paper_id, indices in citation_map.items():
                outcome = enqueue_citation_download(source_paper_id, indices)
                if outcome.get("queued", 0) > 0:
                    if "path A / inline citations" not in triggered_paths:
                        triggered_paths.append("path A / inline citations")
                    status.write(
                        f"Queued background citation download from "
                        f"`{source_paper_id}` for indices {indices}"
                    )

            context = format_context(results)

            history = []
            for message in st.session_state.messages[-6:]:
                if message["role"] in ("user", "assistant") and "sources" not in message:
                    history.append({"role": message["role"], "content": message["content"]})

            status.update(
                label=f"Retrieved {len(results)} papers — generating answer below",
                state="complete",
                expanded=False,
            )

        # Stage 3: stream final answer (no tools). Lives outside the status
        # block so the answer renders as the visible chat message body.
        try:
            generate_answer_stream = get_generate_answer_stream(backend)
            stream = generate_answer_stream(
                question=prompt,
                context=context,
                model=model_name,
                chat_history=history,
            )
            response = st.write_stream(stream)
        except Exception as e:
            if backend == "gemini":
                response = (
                    f"Generation failed: {e}. Is GOOGLE_API_KEY set and is "
                    f"{model_name} a valid Gemini model?"
                )
            else:
                response = f"Generation failed: {e}. Is Ollama running with {model_name}?"
            st.error(response)

        # Stage 4: append download footer when any path triggered.
        footer = _download_attempt_footer(triggered_paths)
        if footer:
            st.markdown(footer)
            response = (response or "") + footer

        sources = [
            {
                "title": result["title"],
                "arxiv_url": result.get("arxiv_url", ""),
                "authors": result.get("authors", ""),
                "passage": result["text"][:300],
                "distance": result["distance"],
            }
            for result in results
        ]
        with st.expander("Sources"):
            for source in sources:
                url = source.get("arxiv_url", "")
                if url:
                    st.markdown(
                        f"**[{source['title']}]({url})** "
                        f"(similarity: {1 - source['distance']:.2%})"
                    )
                else:
                    st.markdown(
                        f"**{source['title']}** "
                        f"(similarity: {1 - source['distance']:.2%})"
                    )
                if source.get("authors"):
                    st.caption(source["authors"])
                st.markdown(
                    f"> {source['passage']}"
                    f"{'...' if len(source['passage']) >= 300 else ''}"
                )
                st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": sources,
            }
        )

    # Force a rerun so the sidebar (Current search range, Pre-RAG debug,
    # Download log) reflects state changed by this turn — Streamlit only
    # re-renders the sidebar on a fresh script execution and the sidebar
    # block runs BEFORE run_query in the script flow.
    st.rerun()


def run_react_query(prompt: str, model_name: str, backend: str) -> None:
    """Execute one ReAct-loop query and append the result to session history.

    Streams ``ReactStep`` events into a live ``st.status`` block so the user
    sees each (thought, action, observation) as the loop runs, then renders
    the final ``ReactFinish.answer`` and accumulated sources just like
    :func:`run_query` does.

    Gemini-only on this branch — Ollama doesn't have a function-calling
    integration story we trust here.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if backend != "gemini":
        with st.chat_message("assistant"):
            err = (
                "ReAct mode requires the Gemini backend on this branch. "
                "Switch backend to `gemini` in the sidebar, or change Query "
                "mode back to `Single-pass`."
            )
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        return

    _wait_for_downloads()

    with st.chat_message("assistant"):
        with st.status("ReAct loop running...", expanded=True) as status:
            final: ReactFinish | None = None
            try:
                for event in run_react(prompt, model=model_name):
                    if isinstance(event, ReactStep):
                        status.update(label=f"Step {event.n}: {event.action}(...)")
                        st.markdown(
                            f"**Step {event.n}** — `{event.action}("
                            f"{', '.join(f'{k}={v!r}' for k, v in event.action_args.items())})`"
                        )
                        if event.thought:
                            st.markdown(f"> _Thought:_ {event.thought}")
                        # Truncate observation in the trace so a 20-chunk
                        # search dump doesn't dominate the UI; the full
                        # context goes to the LLM regardless.
                        obs = event.observation
                        if len(obs) > 1500:
                            obs = obs[:1500].rstrip() + "\n... [truncated]"
                        st.code(obs, language="text")
                    elif isinstance(event, ReactFinish):
                        final = event
                        label_suffix = {
                            "natural": "complete",
                            "max_steps": "complete (max steps reached)",
                            "error": "complete (error)",
                        }.get(event.terminated_by, "complete")
                        status.update(
                            label=f"ReAct loop {label_suffix}",
                            state=(
                                "error" if event.terminated_by == "error"
                                else "complete"
                            ),
                            expanded=False,
                        )
            except Exception as exc:  # noqa: BLE001
                status.update(
                    label=f"ReAct loop failed: {exc}",
                    state="error",
                    expanded=True,
                )
                err = f"ReAct loop failed: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )
                return

        if final is None:
            err = "ReAct loop ended without producing a final answer."
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err}
            )
            return

        st.markdown(final.answer)

        sources = [
            {
                "title": result.get("title", ""),
                "arxiv_url": result.get("arxiv_url", ""),
                "authors": result.get("authors", ""),
                "passage": (result.get("text") or "")[:300],
                "distance": result.get("distance", 0.0),
            }
            for result in final.sources
        ]
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    url = source.get("arxiv_url", "")
                    if url:
                        st.markdown(
                            f"**[{source['title']}]({url})** "
                            f"(similarity: {1 - source['distance']:.2%})"
                        )
                    else:
                        st.markdown(
                            f"**{source['title']}** "
                            f"(similarity: {1 - source['distance']:.2%})"
                        )
                    if source.get("authors"):
                        st.caption(source["authors"])
                    st.markdown(
                        f"> {source['passage']}"
                        f"{'...' if len(source['passage']) >= 300 else ''}"
                    )
                    st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final.answer,
                "sources": sources,
            }
        )

    st.rerun()


def run_recent_summary(prompt: str, top_k: int, recent_days: int, model_name: str,
                       backend: str = "ollama", min_papers: int = 3,
                       max_days_back: int = 30) -> None:
    """Summarize recent papers directly from DB metadata instead of vector search.

    Expanding-window behavior (matches data/get_past_trend.py semantics):
        Start at ``recent_days``. If fewer than ``min_papers`` papers are
        indexed in that window, expand by ``recent_days`` each step
        (e.g. 7 → 14 → 21 → 28) up to ``max_days_back``. Stop at the first
        window that hits the threshold, or fall through to ``max_days_back``
        and use whatever was found.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    _wait_for_downloads()

    # Recent-summary mode does a metadata-only lookup, no vector retrieval —
    # clear the sidebar's retrieval query so it doesn't show a stale value
    # from a previous chat turn.
    retrieval_query_state.clear()

    # Build the expansion ladder once. For recent_days=7, max_days_back=30:
    # [7, 14, 21, 28, 30]. Each entry is a window size we'll try in order.
    windows: list[int] = []
    step = max(1, recent_days)
    w = step
    while w < max_days_back:
        windows.append(w)
        w += step
    if not windows or windows[-1] < max_days_back:
        windows.append(max_days_back)

    with st.chat_message("assistant"):
        with st.status(
            f"Looking up papers from the last {recent_days} days...",
            expanded=True,
        ) as status:
            # retrieve_recent_papers does a metadata-only date filter
            # (collection.get(where=...)) so it can use the lite collection
            # and skip the SentenceTransformer load entirely.
            collection_lite = get_cached_lite_collection()
            results: list[dict] = []
            window_used = windows[-1]
            for idx, window in enumerate(windows):
                try:
                    results = retrieve_recent_papers(
                        recent_days=window,
                        max_papers=top_k,
                        collection=collection_lite,
                    )
                except Exception as e:
                    status.update(label=f"Lookup failed: {e}", state="error", expanded=True)
                    st.error(f"Recent paper lookup failed: {e}. Make sure you've run the ingestion script.")
                    st.stop()

                window_used = window
                if len(results) >= min_papers:
                    break
                # Not enough yet — report and expand (unless we're already at the cap).
                if idx < len(windows) - 1:
                    next_window = windows[idx + 1]
                    status.write(
                        f"Only {len(results)} paper(s) in last {window} days, "
                        f"expanding to {next_window} days..."
                    )

            if not results:
                status.update(
                    label=f"No indexed papers in the last {window_used} days",
                    state="error",
                    expanded=True,
                )
                response = (
                    f"I couldn't find any indexed papers published in the last "
                    f"{window_used} days. Re-run ingestion to fetch newer arXiv papers."
                )
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            status.write(f"Found {len(results)} papers in last {window_used} days")

            context = format_context(results)

            history = []
            for message in st.session_state.messages[-6:]:
                if message["role"] in ("user", "assistant") and "sources" not in message:
                    history.append({"role": message["role"], "content": message["content"]})

            status.update(
                label=f"Found {len(results)} papers in last {window_used} days — generating summary below",
                state="complete",
                expanded=False,
            )

        try:
            generate_answer_stream = get_generate_answer_stream(backend)
            stream = generate_answer_stream(
                question=prompt,
                context=context,
                model=model_name,
                chat_history=history,
            )
            response = st.write_stream(stream)
        except Exception as e:
            if backend == "gemini":
                response = (
                    f"Generation failed: {e}. Is GOOGLE_API_KEY set and is "
                    f"{model_name} a valid Gemini model?"
                )
            else:
                response = f"Generation failed: {e}. Is Ollama running with {model_name}?"
            st.error(response)

        sources = [
            {
                "title": result["title"],
                "arxiv_url": result.get("arxiv_url", ""),
                "authors": result.get("authors", ""),
                "passage": result["text"][:300],
                "distance": result.get("distance", 0.0),
            }
            for result in results
        ]
        with st.expander("Sources"):
            for source in sources:
                url = source.get("arxiv_url", "")
                if url:
                    st.markdown(f"**[{source['title']}]({url})**")
                else:
                    st.markdown(f"**{source['title']}**")
                if source.get("authors"):
                    st.caption(source["authors"])
                st.markdown(
                    f"> {source['passage']}"
                    f"{'...' if len(source['passage']) >= 300 else ''}"
                )
                st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": sources,
            }
        )


st.set_page_config(page_title="CV Paper RAG", page_icon="CV", layout="wide")

# Tighten Streamlit's default top padding (~6rem) on the main content and
# sidebar so the title / Settings header sit closer to the page top.
# Multiple selectors cover Streamlit version differences; !important wins
# over the framework's inline rules.
st.markdown(
    """
    <style>
    /* Make the Streamlit header transparent but keep its height — collapsing
       it shrinks the floating sidebar-reopen button out of the viewport. */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Main content padding */
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewContainer"] .main .block-container,
    .main .block-container {
        padding-top: 1rem !important;
    }

    /* Sidebar reopen button when collapsed: pin to a safe top/left so it
       can't clip above the viewport. Both testids cover Streamlit version
       differences (>=1.27 uses stSidebarCollapsedControl). */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        top: 0.5rem !important;
        left: 0.5rem !important;
        z-index: 999 !important;
    }

    /* Sidebar close-button strip: take it out of normal flow so "Settings"
       can render flush at the top, and pin the close button to top-right
       at a comfortable offset (instead of being jammed against the edge). */
    section[data-testid="stSidebar"] {
        position: relative !important;
    }
    [data-testid="stSidebarHeader"] {
        position: absolute !important;
        top: 0.5rem !important;
        right: 0.5rem !important;
        z-index: 2 !important;
        min-height: 0 !important;
        height: auto !important;
        width: auto !important;
        padding: 0 !important;
    }
    [data-testid="stSidebarHeader"] > div {
        padding: 0 !important;
    }

    /* Sidebar inner padding: zero out so the first heading sits flush with
       the close-button strip. */
    [data-testid="stSidebarNav"] {
        padding: 0 !important;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0 !important;
    }
    /* Pull each non-first sidebar header closer to the divider above it.
       Targets only stElementContainers that (a) contain an h2 AND (b) are
       preceded by an hr-wrapper — so the first "Settings" header (no hr
       before it) and other widgets' inter-spacing are untouched. */
    section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(hr)
        + [data-testid="stElementContainer"]:has(h2) {
        margin-top: -1rem !important;
    }
    section[data-testid="stSidebar"] h2 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] h2:first-of-type {
        margin-top: 1rem !important;
    }

    /* Tighten vertical spacing around sidebar dividers (st.divider()). */
    section[data-testid="stSidebar"] hr,
    section[data-testid="stSidebar"] [data-testid="stDivider"] {
        margin-top: 0rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Pull each divider closer to the widget above it (shrink prev->divider
       gap) by negative-margining its element-container. */
    section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(hr) {
        margin-top: -0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Ask me anything about computer vision research papers!",
        }
    ]
if "time_range" not in st.session_state:
    st.session_state["time_range"] = {"start_date": None, "end_date": None}
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "pending_top_k" not in st.session_state:
    st.session_state["pending_top_k"] = None
if "pending_mode" not in st.session_state:
    st.session_state["pending_mode"] = "chat"

_ts("about to render sidebar")

with st.sidebar:
    st.header("Settings")
    query_mode = st.selectbox(
        "Query mode",
        ["Single-pass", "ReAct"],
        index=0,
        help=(
            "Single-pass: one Gemini pre-RAG pass extracts intent (date "
            "filters, citation downloads) and one retrieve() call feeds the "
            "answer. ReAct: bounded multi-step loop where Gemini chooses "
            "search_papers / list_paper_titles / set_time_range across up "
            "to 3 rounds before answering. Gemini backend only."
        ),
    )
    top_k = st.slider("Number of retrieved chunks", 1, 20, TOP_K)
    use_reranker = st.toggle(
        "Cross-encoder reranker",
        value=True,
        help=(
            "Fetches a larger candidate pool from ChromaDB and re-ranks it "
            "with a cross-encoder before truncating to top-k. Improves "
            "ranking quality at the cost of ~50ms per query."
        ),
    )
    if st.button("Summarize new papers from last 7 days", use_container_width=True):
        st.session_state["pending_prompt"] = WEEKLY_SUMMARY_PROMPT
        st.session_state["pending_top_k"] = max(top_k, 12)
        st.session_state["pending_mode"] = "recent_summary"
        # Reflect the 7-day window in the sidebar's "Current search range"
        # without going through Gemini — this is a deterministic preset, not
        # a natural-language request that needs intent extraction.
        _today = datetime.date.today()
        st.session_state["time_range"] = {
            "start_date": (_today - datetime.timedelta(days=7)).isoformat(),
            "end_date": _today.isoformat(),
        }

    st.divider()
    if generator_gemini.last_pre_rag_error:
        st.error(generator_gemini.last_pre_rag_error)
    st.header("Retrieval query")
    if retrieval_query_state.cleaned:
        st.code(retrieval_query_state.cleaned, language="text")
    else:
        st.write("No query yet.")
    st.caption(
        "The query string actually sent to ChromaDB and the reranker. "
        "Function/tool `rewrite_retrieval_query` tool may strip "
        "date filters and conversational filler from the raw prompt"
    )

    st.divider()
    st.header("Current search range")
    st.write(_format_time_range_display())
    st.caption(
        "Set automatically when you mention dates in chat (Gemini backend). "
        "Say \"ignore the time filter\" to clear."
    )

    st.divider()
    st.header("Download log")

    # Capture busy state at script-render time. The fragment below uses
    # this to decide whether to enable polling. Once polling reaches a
    # busy → idle transition it forces a full script rerun so the next
    # fragment instantiation drops ``run_every`` and stops ticking.
    _busy_at_render = is_busy()

    @st.fragment(run_every="2s" if _busy_at_render else None)
    def _download_log_fragment() -> None:
        # Consume the worker→main-thread dirty flag. The flag itself is
        # mostly informational here because Streamlit's WebSocket layer
        # diffs the rendered DOM and only sends actual changes — but
        # clearing it documents the signal handoff and prevents the flag
        # from accumulating "stale set" state across reruns.
        take_log_change_signal()

        log_entries = list(reversed(get_log_snapshot()))
        if not log_entries:
            st.caption("No background downloads yet.")
        else:
            for entry in log_entries:
                icon = {
                    "ok": "[OK]",
                    "failed": "[FAIL]",
                    "skipped": "[SKIP]",
                    "running": "[...]",
                    "pending": "[...]",
                }.get(entry.status, "[?]")
                # Prefer the human-readable title once it's resolved; fall
                # back to arxiv_id (which is the cite_label for pending /
                # pre-resolve rows).
                display = entry.title or entry.arxiv_id
                line = f"{icon} `{display}`"
                if entry.reason:
                    line += f" — {entry.reason}"
                st.markdown(line)

        currently_busy = is_busy()
        if currently_busy:
            st.caption("Background ingestion in progress...")
        elif _busy_at_render:
            # Transition: fragment was polling, now idle. Full-script
            # rerun re-evaluates ``_busy_at_render`` so the next fragment
            # instance has ``run_every=None`` and stops ticking.
            # Brief sleep gives any race-finishing workers time to flush
            # their last log entry before we settle.
            time.sleep(0.3)
            st.rerun()

    _download_log_fragment()

    st.divider()
    backend_options = ["ollama", "gemini"]
    default_backend_index = backend_options.index(LLM_BACKEND) if LLM_BACKEND in backend_options else 0
    backend = st.selectbox(
        "LLM backend",
        backend_options,
        index=default_backend_index,
        help="ollama = local/VM Ollama (original). gemini = Google AI Studio Gemini free tier.",
    )
    if backend == "gemini":
        model_name = st.text_input("Gemini model", value=GEMINI_MODEL)
        st.caption(
            "Set GOOGLE_API_KEY env var. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    else:
        model_name = st.text_input("Ollama model", value=OLLAMA_MODEL)

    st.divider()
    with st.expander("Last turn function calls", expanded=False):
        if rag_tools.last_call_log:
            for line in rag_tools.last_call_log:
                st.code(line, language="text")
        elif backend == "gemini" and not generator_gemini.last_pre_rag_error:
            st.caption("No tool calls fired in the last turn.")
        else:
            st.caption("(Gemini backend only)")

    with st.expander("Collection Stats", expanded=False):
        try:
            _ts("  before get_cached_chunk_count()")
            _count = get_cached_chunk_count()
            _ts(f"  after get_cached_chunk_count() = {_count}")
            st.metric("Indexed chunks", _count)
        except Exception:
            st.warning("No indexed data yet. Run the ingestion script first.")

_ts("sidebar rendered, about to render main area")

st.title("Computer Vision Paper RAG")

if backend == "gemini":
    st.caption(f"Powered by {model_name} via Google AI Studio + ChromaDB retrieval")
else:
    st.caption(f"Powered by {model_name} via Ollama + ChromaDB retrieval")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for source in msg["sources"]:
                    title = source.get("title", "Unknown")
                    url = source.get("arxiv_url", "")
                    authors = source.get("authors", "")
                    passage = source.get("passage", "")
                    distance = source.get("distance", 0)
                    if url:
                        st.markdown(f"**[{title}]({url})** (similarity: {1 - distance:.2%})")
                    else:
                        st.markdown(f"**{title}** (similarity: {1 - distance:.2%})")
                    if authors:
                        st.caption(authors)
                    if passage:
                        st.markdown(f"> {passage[:200]}{'...' if len(passage) > 200 else ''}")

prompt = st.chat_input("Ask a question about computer vision papers...")
active_prompt = st.session_state.pop("pending_prompt", None) or prompt
active_top_k = st.session_state.pop("pending_top_k", None) or top_k
active_mode = st.session_state.pop("pending_mode", "chat")

_ts("main area rendered, script body finished")

if active_prompt:
    if active_mode == "recent_summary":
        # The 7-day-summary preset always runs the metadata-only path,
        # regardless of whether the sidebar is set to ReAct mode.
        run_recent_summary(
            active_prompt,
            top_k=active_top_k,
            recent_days=7,
            model_name=model_name,
            backend=backend,
        )
    elif query_mode == "ReAct":
        run_react_query(
            active_prompt,
            model_name=model_name,
            backend=backend,
        )
    else:
        run_query(
            active_prompt,
            top_k=active_top_k,
            model_name=model_name,
            backend=backend,
            use_reranker=use_reranker,
        )
