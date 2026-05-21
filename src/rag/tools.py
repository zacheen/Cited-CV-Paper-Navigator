"""Gemini Native Function Calling tools for the RAG chatbot.

These three functions are passed to ``GenerateContentConfig(tools=[...])``
during the pre-RAG pass. Gemini's Automatic Function Calling introspects
their signatures and docstrings to decide when to invoke them.

State is held at module level so the tools stay pure-Python (no Streamlit
imports). ``app.py`` syncs ``st.session_state["time_range"]`` to and from the
:data:`time_range_state` object across each chat turn.

NOTE: deliberately NO ``from __future__ import annotations`` here. Gemini's
AFC uses ``inspect.signature`` to build the function schema, and PEP 563
deferred annotations turn the type hints into raw strings that the SDK
fails to introspect — silently disabling auto-execution of the tool.
"""

import datetime
import sys
from dataclasses import dataclass, field
from typing import Callable, List

from src.rag.download_state import enqueue_citation_download

# Optional collection provider, injected by app.py so the ReAct tools reuse
# the cached SentenceTransformer-backed collection instead of paying the
# ~17s cold load on every CLI invocation. CLI / test usage works without
# this — :func:`_get_collection` falls back to a fresh :func:`get_collection`
# call.
_collection_provider: "Callable | None" = None


def set_collection_provider(provider: Callable) -> None:
    """Inject a Streamlit-cached collection getter for the ReAct tools.

    Called by ``app.py`` at module load. The provider is a zero-arg callable
    returning a ChromaDB ``Collection``.
    """
    global _collection_provider
    _collection_provider = provider


def _get_collection():
    """Return the (possibly cached) collection used by ReAct retrieval tools."""
    if _collection_provider is not None:
        return _collection_provider()
    from src.processing.embedder import get_collection
    return get_collection()


@dataclass
class TimeRangeState:
    """Module-level mirror of the active publication-date filter."""

    start_date: "str | None" = None
    end_date: "str | None" = None

    def clear(self) -> None:
        self.start_date = None
        self.end_date = None

    def set(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date

    def to_dict(self) -> dict:
        return {"start_date": self.start_date, "end_date": self.end_date}


time_range_state = TimeRangeState()


@dataclass
class RetrievalQueryState:
    """Per-turn cleaned query for vector retrieval. Cleared at the start of
    every pre-RAG pass; never persisted across turns (each user message gets
    a fresh chance to be rewritten)."""

    cleaned: "str | None" = None

    def clear(self) -> None:
        self.cleaned = None

    def set(self, cleaned: str) -> None:
        self.cleaned = cleaned


retrieval_query_state = RetrievalQueryState()

# Debug visibility: every tool invocation appends a one-line summary here.
# Cleared by run_pre_rag_pass at the start of each turn. The sidebar reads
# this to show what the LLM actually called (or didn't call).
last_call_log: list[str] = []


def _validate_iso_date(value: str, field_name: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be an ISO date YYYY-MM-DD, got {value!r}"
        ) from exc


def set_time_range(start_date: str, end_date: str) -> str:
    """Set the publication-date range used to filter retrieved papers.

    Call this whenever the user wants to constrain results to papers
    published within a specific window. The LLM is responsible for
    converting natural-language phrases ("last month", "March 2025",
    "since 2024") into absolute ISO dates using today's date provided in
    the system prompt. Both dates are inclusive.

    Args:
        start_date: ISO date string YYYY-MM-DD (inclusive).
        end_date: ISO date string YYYY-MM-DD (inclusive). If the user did
            not specify an end, pass today's date.

    Returns:
        Human-readable confirmation string.
    """
    print(f"[TOOL] set_time_range({start_date!r}, {end_date!r})", file=sys.stderr, flush=True)
    last_call_log.append(f"set_time_range({start_date!r}, {end_date!r})")
    try:
        start = _validate_iso_date(start_date, "start_date")
        end = _validate_iso_date(end_date, "end_date")
    except ValueError as exc:
        return f"Error: {exc}"

    if start > end:
        return (
            f"Error: start_date {start_date} is after end_date {end_date}; "
            "no change applied."
        )

    time_range_state.set(start.isoformat(), end.isoformat())
    return f"Time range set to {start.isoformat()} -> {end.isoformat()}."


def clear_time_range() -> str:
    """Remove the publication-date filter so retrieval covers all indexed papers.

    Call this when the user explicitly asks to ignore date constraints
    (for example: "search all papers", "no date filter", "ignore the time
    range").

    Returns:
        Confirmation string.
    """
    print("[TOOL] clear_time_range()", file=sys.stderr, flush=True)
    last_call_log.append("clear_time_range()")
    time_range_state.clear()
    return "Time range cleared. Searching all indexed papers."


def download_cited_papers(
    source_paper_id: str, citation_indices: List[int]
) -> str:
    """Download papers cited by a specific source paper.

    Use this ONLY when the user explicitly asks to fetch the references
    cited by a particular paper that is NOT already shown in the retrieved
    context (for example: "download all papers cited by arXiv:2103.12345",
    or "fetch the references of the ViT paper" when ViT is not in our
    current retrieval). The application automatically downloads citations
    detected in the retrieved context via a deterministic regex path —
    do not call this tool for those cases.

    Args:
        source_paper_id: arXiv id of the source paper containing the
            citations (e.g. "2103.12345").
        citation_indices: 1-indexed citation numbers as they appear in the
            source paper's References section (e.g. [1, 5, 12]).

    Returns:
        Status string. Actual download and ingestion happen asynchronously;
        the user can monitor progress in the sidebar download log.
    """
    print(
        f"[TOOL] download_cited_papers({source_paper_id!r}, {list(citation_indices)!r})",
        file=sys.stderr,
        flush=True,
    )
    last_call_log.append(
        f"download_cited_papers({source_paper_id!r}, {list(citation_indices)!r})"
    )
    if not source_paper_id or not citation_indices:
        return "Error: source_paper_id and citation_indices are both required."

    try:
        indices = [int(i) for i in citation_indices]
    except (TypeError, ValueError):
        return "Error: citation_indices must be a list of integers."

    outcome = enqueue_citation_download(source_paper_id, indices)
    queued = outcome.get("queued", 0)
    skipped = outcome.get("skipped", 0)
    if queued == 0 and skipped > 0:
        return (
            f"No new download queued for {source_paper_id}: "
            f"{outcome.get('reason', 'duplicate request')}."
        )
    return (
        f"Queued background download of {queued} reference(s) cited by "
        f"{source_paper_id} (indices: {indices}). Check the sidebar "
        "download log for results."
    )


def rewrite_retrieval_query(cleaned_query: str) -> str:
    """Provide a cleaner version of the user's prompt to use as the
    semantic-search query for paper retrieval.

    Call this whenever the user's prompt contains material that doesn't help
    vector retrieval — e.g. time-range phrases ("last week", "in 2025",
    "this year"), download requests ("download the references of..."),
    pleasantries, instructions to the chatbot. Strip those and keep ONLY the
    topical keywords describing what the user wants to learn about.

    Examples:
      User: "I want the diffusion application within this year"
        -> cleaned_query = "diffusion model applications"
      User: "what's new with vision transformers since 2024?"
        -> cleaned_query = "vision transformers"
      User: "download papers cited by ViT and explain self-attention"
        -> cleaned_query = "self-attention in vision transformers"

    Skip this tool only when the user's prompt is already clean topical
    keywords with no noise to strip.

    Args:
        cleaned_query: Concise topical query (5-20 words) suitable for
            sentence-embedding similarity search. Do NOT include dates,
            download instructions, or conversational phrasing.

    Returns:
        Confirmation string.
    """
    print(
        f"[TOOL] rewrite_retrieval_query({cleaned_query!r})",
        file=sys.stderr,
        flush=True,
    )
    last_call_log.append(f"rewrite_retrieval_query({cleaned_query!r})")
    cleaned = (cleaned_query or "").strip()
    if not cleaned:
        return "Error: cleaned_query cannot be empty."
    retrieval_query_state.set(cleaned)
    return f"Retrieval query set to: {cleaned}"


def get_tools() -> list:
    """Return the list of tool callables to register with Gemini AFC."""
    return [
        set_time_range,
        clear_time_range,
        download_cited_papers,
        rewrite_retrieval_query,
    ]


# =============================================================================
# ReAct mode tools and state
# =============================================================================
#
# The ReAct agent runs a manual loop where the LLM picks a tool each round
# until it stops calling tools (text-only response = final answer). Tools
# below are the ReAct-mode tool set (NOT used by the single-pass pre-RAG
# flow above):
#
#   - search_papers       : semantic search; primary information-gathering
#   - list_paper_titles   : metadata-only listing for "what's recent"
#   - set_time_range      : reused from pre-RAG
#   - clear_time_range    : reused from pre-RAG
#
# Module-level state below tracks (a) the chunks accumulated across all
# search_papers calls in the current turn (so the final answer's "Sources"
# expander can cite them) and (b) the recent search-query history (for the
# anti-loop guard in react_agent).


@dataclass
class ReactRetrievalState:
    """Accumulates retrieved chunks across all search_papers calls in a turn.

    The ReAct agent may call search_papers several times with different
    queries; the final-answer UI shows the union of all chunks the LLM
    actually saw, deduped by (paper_id, chunk_index).
    """

    chunks: list = field(default_factory=list)

    def add(self, new_chunks: list) -> None:
        seen = {(c.get("paper_id"), c.get("chunk_index")) for c in self.chunks}
        for c in new_chunks:
            key = (c.get("paper_id"), c.get("chunk_index"))
            if key not in seen:
                self.chunks.append(c)
                seen.add(key)

    def clear(self) -> None:
        self.chunks.clear()


react_retrieval_state = ReactRetrievalState()


@dataclass
class ReactSearchState:
    """Tracks normalized queries already searched in this turn.

    Anti-loop guard: if the LLM asks search_papers with the same normalized
    query as a recent step, react_agent short-circuits with a synthetic
    observation telling the model to vary its angle, instead of re-running
    the (expensive) retrieval and pumping the same chunks back.
    """

    queries: list = field(default_factory=list)

    def normalize(self, q: str) -> str:
        return " ".join((q or "").lower().split())

    def has_seen(self, q: str) -> bool:
        return self.normalize(q) in self.queries

    def record(self, q: str) -> None:
        n = self.normalize(q)
        if n and n not in self.queries:
            self.queries.append(n)

    def clear(self) -> None:
        self.queries.clear()


react_search_state = ReactSearchState()


def _format_search_observation(query: str, chunks: list) -> str:
    """Format retrieved chunks into a compact observation string for the LLM.

    Goal: maximize information per token. The LLM doesn't need full chunk
    text in its conversation history — paper id, title, and a short preview
    are usually enough to decide whether to search further or to answer.
    The full chunk text is still in :data:`react_retrieval_state.chunks` if
    the model wants to use it; it'll be included in the final-answer
    context via :func:`format_context`.
    """
    if not chunks:
        return f"No results for query {query!r}."
    lines = [f"Found {len(chunks)} chunks for query {query!r}:"]
    for i, c in enumerate(chunks, 1):
        title = (c.get("title") or "").strip() or "Untitled"
        paper_id = (c.get("paper_id") or "").strip() or "unknown-id"
        preview = (c.get("text") or "").strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220].rstrip() + "..."
        lines.append(f"[{i}] {title} (arxiv:{paper_id})")
        lines.append(f"    {preview}")
    return "\n".join(lines)


def search_papers(query: str, k: int = 5) -> str:
    """Search the indexed computer-vision paper corpus by semantic similarity.

    Retrieves the top-k most relevant chunks for ``query`` from ChromaDB,
    then re-ranks them with a cross-encoder for sharper ordering. Honors
    any active publication-date filter set via :func:`set_time_range`.

    Each call accumulates its results into the turn's source pool, so the
    final answer can cite chunks from multiple distinct searches. Use
    different queries across calls to cover multiple angles — repeating the
    same query is detected upstream and short-circuited.

    Args:
        query: Topical search string (5-20 words). Use keywords describing
            what you want to learn, not full natural-language questions.
        k: Number of chunks to return (1-10, default 5).

    Returns:
        Multi-line summary listing each chunk's paper title, arXiv id, and
        a short preview. Use this to decide whether to search again or to
        write the final answer.
    """
    print(f"[TOOL] search_papers({query!r}, k={k})", file=sys.stderr, flush=True)
    last_call_log.append(f"search_papers({query!r}, k={k})")

    cleaned = (query or "").strip()
    if not cleaned:
        return "Error: query cannot be empty."
    k = max(1, min(int(k), 10))

    # Defer import to call time so tools.py doesn't pay sentence-transformers
    # import cost when only the simpler tools are in play.
    from src.rag.retriever import retrieve

    start = (
        datetime.date.fromisoformat(time_range_state.start_date)
        if time_range_state.start_date
        else None
    )
    end = (
        datetime.date.fromisoformat(time_range_state.end_date)
        if time_range_state.end_date
        else None
    )

    try:
        chunks = retrieve(
            cleaned,
            top_k=k,
            collection=_get_collection(),
            start_date=start,
            end_date=end,
            use_reranker=True,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: retrieval failed: {exc}"

    react_retrieval_state.add(chunks)
    return _format_search_observation(cleaned, chunks)


def list_paper_titles(days_back: int = 7, limit: int = 20) -> str:
    """List paper titles indexed within the last N days, by metadata only.

    Use this BEFORE search_papers when the user asks "what's new", "any
    recent work on X", or anything date-scoped — listing titles is much
    cheaper than a vector search and gives the agent a sense of what's in
    the corpus before it commits to a query string. Does NOT require an
    active time range.

    Args:
        days_back: Look back this many days from today (1-60, default 7).
        limit: Maximum number of papers to return (1-50, default 20).

    Returns:
        Newline-separated list of "title (arxiv:id) — published_date".
    """
    print(
        f"[TOOL] list_paper_titles(days_back={days_back}, limit={limit})",
        file=sys.stderr,
        flush=True,
    )
    last_call_log.append(f"list_paper_titles(days_back={days_back}, limit={limit})")

    days_back = max(1, min(int(days_back), 60))
    limit = max(1, min(int(limit), 50))

    # retrieve_recent_papers does the metadata-only date filter and
    # one-row-per-paper dedup we want here. Reuse it instead of re-doing.
    from src.processing.embedder import get_collection_lite
    from src.rag.retriever import retrieve_recent_papers

    try:
        papers = retrieve_recent_papers(
            recent_days=days_back,
            max_papers=limit,
            collection=get_collection_lite(),
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: title lookup failed: {exc}"

    if not papers:
        return f"No papers indexed in the last {days_back} days."

    lines = [f"Found {len(papers)} papers in the last {days_back} days:"]
    for p in papers:
        title = (p.get("title") or "").strip() or "Untitled"
        paper_id = (p.get("paper_id") or "").strip() or "unknown-id"
        published = (p.get("published") or "").strip()[:10] or "unknown-date"
        lines.append(f"- {title} (arxiv:{paper_id}) — {published}")
    return "\n".join(lines)


def get_react_tools() -> list:
    """Return the list of tool callables for ReAct mode.

    Differs from :func:`get_tools` (single-pass pre-RAG): includes the
    primary search/list tools and date-range setters, but excludes
    download_cited_papers (kept as a single-pass-only special action) and
    rewrite_retrieval_query (the ReAct agent picks query strings directly,
    no rewrite pass needed).
    """
    return [
        search_papers,
        list_paper_titles,
        set_time_range,
        clear_time_range,
    ]
