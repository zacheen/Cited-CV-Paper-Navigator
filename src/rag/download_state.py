"""Background download/ingest job queue and dedup helpers.

Shared by both trigger paths for cited-paper download:

- Path A (regex auto): app.py extracts inline ``[X]`` citations from
  retrieved chunks via :func:`extract_inline_citations` and feeds each
  ``(source_paper_id, indices)`` pair to :func:`enqueue_citation_download`.
- Path B (LLM tool): the ``download_cited_papers`` tool calls the same
  :func:`enqueue_citation_download` directly when the user explicitly asks.

All state lives at module level so background threads can mutate it without
touching Streamlit's session state. The Streamlit UI reads
:func:`get_log_snapshot` / :func:`is_busy` on each rerun.
"""

from __future__ import annotations

import datetime
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.config import PDF_DIR
from src.processing.embedder import get_collection_lite

if TYPE_CHECKING:
    import chromadb

LOG_MAX_ENTRIES = 100
INLINE_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Optional DI hook: long-running hosts (the Streamlit app) inject a cached
# lite-collection factory here so the metadata-only DB checks reuse a single
# ChromaDB client across reruns and background threads. Scripts/tests don't
# set this and fall back to building a fresh lite collection on each call.
#
# Note: the only collection access in this module is ``_is_in_db`` which does
# ``collection.get(where=...)`` — purely metadata. We deliberately use the
# lite (no-embedder) variant here so background dedup checks never trigger
# the SentenceTransformer load.
_lite_collection_provider: Callable[[], "chromadb.Collection"] | None = None


def set_lite_collection_provider(
    provider: Callable[[], "chromadb.Collection"] | None,
) -> None:
    """Install a process-wide lite (no-embedder) collection factory.

    Pass ``None`` to reset to the default (build a fresh lite collection on
    each call). Kept as a module-level injection point so this module stays
    free of Streamlit-specific imports.
    """
    global _lite_collection_provider
    _lite_collection_provider = provider


def _resolve_lite_collection() -> "chromadb.Collection":
    if _lite_collection_provider is not None:
        return _lite_collection_provider()
    return get_collection_lite()


@dataclass
class DownloadJobEntry:
    arxiv_id: str  # display id (cite_label initially, resolved id once known)
    status: str  # "pending" | "running" | "ok" | "failed" | "skipped"
    reason: str = ""
    started_at: str = ""
    finished_at: str = ""
    source_paper_id: str = ""
    # Stable update key for upsert. Set to the cite_label
    # (``{source}#[{idx}]``) for citation-driven jobs so the worker can
    # transition pending → running → ok/failed without losing the original
    # row even if the displayed ``arxiv_id`` changes when a real arXiv id
    # gets resolved. Empty for non-citation entries (those upsert by
    # ``arxiv_id``).
    cite_label: str = ""
    # Resolved paper title from Semantic Scholar (or arXiv title-search).
    # Empty until the reference list resolves; the sidebar prefers this over
    # ``arxiv_id`` for human-readable display once it's set.
    title: str = ""


_lock = threading.Lock()
_log: deque[DownloadJobEntry] = deque(maxlen=LOG_MAX_ENTRIES)
_in_flight: set[str] = set()  # arxiv_ids currently pending or running
_busy_count: int = 0  # number of background workers currently running

# Worker → main-thread signal. Workers set this whenever the log changes;
# the Streamlit UI polls/consumes it to decide when to re-render the
# Download log section. Independent of ``_busy_count`` so log edits that
# don't change busy-ness (e.g. status flipping ``running → ok``) still
# notify.
_log_changed: threading.Event = threading.Event()


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _upsert_log(entry: DownloadJobEntry) -> None:
    """Insert or update a log entry, keyed by ``cite_label`` when set,
    otherwise ``arxiv_id``. Lets workers transition a pre-populated
    pending row through running → ok/failed without spawning duplicates.
    """
    key = entry.cite_label or entry.arxiv_id
    with _lock:
        for i, existing in enumerate(_log):
            existing_key = existing.cite_label or existing.arxiv_id
            if existing_key == key:
                _log[i] = entry
                break
        else:
            _log.append(entry)
    _log_changed.set()


# Back-compat shim: callers that just want to append unconditionally still
# work, but go through the same notify path. New code should use
# :func:`_upsert_log` so pre-populated entries get updated in place.
def _push_log(entry: DownloadJobEntry) -> None:
    _upsert_log(entry)


def take_log_change_signal() -> bool:
    """Test-and-clear the log-changed flag atomically.

    Returns True if a worker pushed a new log entry since the last call.
    Use this as the reactive gate from the Streamlit UI: combine with a
    short-interval poll (e.g. ``st.fragment(run_every="2s")``) so the
    sidebar re-renders only when something actually changed.
    """
    if _log_changed.is_set():
        _log_changed.clear()
        return True
    return False


def get_log_snapshot() -> list[DownloadJobEntry]:
    """Return a copy of the current download log (newest last)."""
    with _lock:
        return list(_log)


def is_busy() -> bool:
    """True iff a short-running background job is currently in flight.

    Covers citation downloads (typically ~30s for 1-5 papers). Does **not**
    cover the long-running past-trend ingestion job — that has its own
    :func:`is_past_trend_running` flag because conflating the two would
    force unrelated UI paths (e.g. ``_wait_for_downloads``) to block for
    10+ minutes during a past-trend run.
    """
    with _lock:
        return _busy_count > 0


def _is_in_db(arxiv_id: str) -> bool:
    """Check whether ChromaDB already has at least one chunk for arxiv_id."""
    try:
        collection = _resolve_lite_collection()
        # paper_id in chunk metadata uses the arXiv-id-with-slashes form.
        result = collection.get(where={"paper_id": arxiv_id}, limit=1)
        return bool(result and result.get("ids"))
    except Exception:
        return False


def _pdf_path_for(arxiv_id: str) -> Path:
    return PDF_DIR / f"{_safe_id(arxiv_id)}.pdf"


def _gate_check(arxiv_id: str) -> tuple[bool, str]:
    """Return (should_enqueue, reason).

    Runs the dedup gates in order:
      1. ChromaDB metadata
      2. PDF file already on disk
      3. Already in the in-flight set
    """
    if _is_in_db(arxiv_id):
        return False, "already in DB"
    if _pdf_path_for(arxiv_id).exists():
        # PDF exists but not in DB → still enqueue so the worker can re-ingest.
        return True, "re-index existing pdf"
    with _lock:
        if arxiv_id in _in_flight:
            return False, "in flight"
    return True, ""


def enqueue_citation_download(
    source_paper_id: str,
    citation_indices: list[int],
    *,
    worker: Callable[[str, list[int]], None] | None = None,
) -> dict:
    """Enqueue a background job that resolves citations and ingests papers.

    The worker is dispatched as a daemon thread and:
      1. Resolves ``citation_indices`` against the source paper's references
         via Semantic Scholar (handled inside the worker).
      2. For each resolved arXiv id, re-runs the dedup gates and downloads +
         ingests if still needed.
      3. Updates the module-level log + busy counter.

    Args:
        source_paper_id: arXiv id of the paper containing the citations.
        citation_indices: 1-indexed citation numbers as they appear in the
            source paper's text.
        worker: Optional override (used by tests). Defaults to the real worker
            in :mod:`src.rag.download_worker` (resolved lazily to avoid an
            import cycle).

    Returns:
        Dict summarizing the enqueue outcome.
    """
    citation_indices = sorted(set(int(i) for i in citation_indices))
    if not source_paper_id or not citation_indices:
        return {"queued": 0, "skipped": 0, "reason": "empty input"}

    if worker is None:
        from src.rag.download_worker import run_citation_download_job

        worker = run_citation_download_job

    job_key = f"{source_paper_id}::{','.join(str(i) for i in citation_indices)}"
    with _lock:
        if job_key in _in_flight:
            return {"queued": 0, "skipped": len(citation_indices), "reason": "job in flight"}
        _in_flight.add(job_key)
        global _busy_count
        _busy_count += 1

    # Show the full work list to the user up-front. The worker will update
    # each row in place as it progresses.
    prepopulate_pending_entries(source_paper_id, citation_indices)

    def _run() -> None:
        try:
            worker(source_paper_id, citation_indices)
        except Exception as exc:  # noqa: BLE001
            # Safety net: any unhandled exception inside the worker still
            # surfaces in the Download log (and trips ``_log_changed``) so
            # the UI's polling fragment notices the failure instead of
            # leaving stale ``running`` entries forever.
            record_result(
                arxiv_id=f"{source_paper_id}#error",
                status="failed",
                reason=f"worker crashed: {exc!r}",
                source_paper_id=source_paper_id,
            )
        finally:
            with _lock:
                _in_flight.discard(job_key)
                global _busy_count
                _busy_count -= 1

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "queued": len(citation_indices),
        "skipped": 0,
        "source_paper_id": source_paper_id,
    }


# ----------------------------------------------------------------------
# Past-trend (HF Daily CV ingestion) background job
# ----------------------------------------------------------------------
# Used by the Streamlit "Summarize new papers" button: after the synchronous
# summary completes, kick off a background `run_past_trend()` so the next
# click sees fresher data. Lives in the same module as citation downloads
# because it shares _busy_count / record_result / threading semantics —
# _wait_for_downloads() in app.py blocks on both via is_busy().

_past_trend_lock = threading.Lock()
_past_trend_in_flight: bool = False
_past_trend_last_completed_at: float = 0.0  # monotonic time of last finish

# Don't re-trigger if a job finished within this many seconds — the data is
# already as fresh as the previous run could make it. Prevents rapid clicking
# from hammering arXiv + HF unnecessarily.
_PAST_TREND_FRESHNESS_WINDOW_SEC = 30 * 60  # 30 minutes


def is_past_trend_running() -> bool:
    """True iff a background past-trend ingestion job is currently in flight."""
    with _past_trend_lock:
        return _past_trend_in_flight


def enqueue_past_trend_download(
    days_back: int = 7,
    min_new_papers: int | None = 3,
    max_days_back: int = 30,
    stop_on_indexed: bool = True,
) -> bool:
    """Kick off a background past-trend ingestion thread.

    Skips (returns False) if a job is already running, or if one completed
    within the freshness window. Returns True if a new thread was started.

    Args mirror :func:`src.ingestion.past_trend.run_past_trend`. Default
    ``stop_on_indexed=True`` matches the past-trend CLI semantics — walk
    backwards filling the corpus until hitting already-synced ground.
    """
    global _past_trend_in_flight, _past_trend_last_completed_at, _busy_count

    with _past_trend_lock:
        if _past_trend_in_flight:
            return False
        if _past_trend_last_completed_at > 0:
            since = time.monotonic() - _past_trend_last_completed_at
            if since < _PAST_TREND_FRESHNESS_WINDOW_SEC:
                return False
        _past_trend_in_flight = True

    # Deliberately NOT incrementing _busy_count — past_trend can run for 10+
    # minutes and conflating it with citation downloads forces every call to
    # _wait_for_downloads() (which polls is_busy()) to block for that long.
    # Callers that specifically care about past_trend status use
    # is_past_trend_running() instead.

    record_result(
        arxiv_id="recent_paper_ingestion",
        status="running",
        reason=f"Fetching recent CV papers (window: {days_back}-{max_days_back} days)",
    )

    def _run() -> None:
        global _past_trend_in_flight, _past_trend_last_completed_at
        try:
            # Lazy import: keeps download_state import-light and avoids any
            # circular import at module load time (past_trend pulls in
            # embedder/chunker which are heavier).
            from src.ingestion.past_trend import run_past_trend  # noqa: PLC0415

            run_past_trend(
                days_back=days_back,
                min_new_papers=min_new_papers,
                stop_on_indexed=stop_on_indexed,
                max_days_back=max_days_back,
            )
            record_result(
                arxiv_id="recent_paper_ingestion",
                status="ok",
                reason="Recent paper ingestion complete",
            )
        except Exception as exc:  # noqa: BLE001
            record_result(
                arxiv_id="recent_paper_ingestion",
                status="failed",
                reason=f"Past-trend worker crashed: {exc!r}",
            )
        finally:
            with _past_trend_lock:
                _past_trend_in_flight = False
                _past_trend_last_completed_at = time.monotonic()

    thread = threading.Thread(target=_run, daemon=True, name="past-trend-worker")
    thread.start()
    return True


def mark_arxiv_in_flight(arxiv_id: str) -> bool:
    """Reserve an arxiv_id slot. Returns False if already in flight."""
    with _lock:
        if arxiv_id in _in_flight:
            return False
        _in_flight.add(arxiv_id)
        return True


def release_arxiv(arxiv_id: str) -> None:
    with _lock:
        _in_flight.discard(arxiv_id)


def record_result(
    arxiv_id: str,
    status: str,
    reason: str = "",
    source_paper_id: str = "",
    started_at: str = "",
    cite_label: str = "",
    title: str = "",
) -> None:
    """Insert or update a log entry.

    When ``cite_label`` is supplied (citation-driven jobs), the row is
    matched by cite_label so a pre-populated pending entry transitions
    in place rather than producing a duplicate. Without ``cite_label``
    the row is matched by ``arxiv_id`` (back-compat for callers that
    don't track citation indices).
    """
    entry = DownloadJobEntry(
        arxiv_id=arxiv_id,
        status=status,
        reason=reason,
        started_at=started_at or _now_iso(),
        finished_at=_now_iso(),
        source_paper_id=source_paper_id,
        cite_label=cite_label,
        title=title,
    )
    _upsert_log(entry)


def prepopulate_pending_entries(
    source_paper_id: str, citation_indices: list[int]
) -> None:
    """Push a "pending" row per requested citation so the sidebar shows
    the full work list immediately, before the background worker starts
    resolving and downloading. Each row is keyed by ``cite_label`` and
    later updated in place by :func:`record_result`.
    """
    started = _now_iso()
    for idx in citation_indices:
        cite_label = f"{source_paper_id}#[{idx}]"
        entry = DownloadJobEntry(
            arxiv_id=cite_label,
            status="pending",
            reason="",
            started_at=started,
            finished_at="",
            source_paper_id=source_paper_id,
            cite_label=cite_label,
        )
        _upsert_log(entry)


def gate_check(arxiv_id: str) -> tuple[bool, str]:
    """Public wrapper around the dedup gates for use by workers."""
    return _gate_check(arxiv_id)


def now_iso() -> str:
    return _now_iso()


def extract_inline_citations(retrieved_chunks: list[dict]) -> dict[str, list[int]]:
    """Scan retrieved chunks for ``[N]`` and ``[N, M]`` citation patterns.

    Args:
        retrieved_chunks: List of dicts with at least ``paper_id`` and ``text``
            keys (the same shape returned by :func:`src.rag.retriever.retrieve`).

    Returns:
        Mapping ``{source_paper_id: [sorted unique citation indices]}``.
        Chunks without a ``paper_id`` are skipped.
    """
    result: dict[str, set[int]] = {}
    for chunk in retrieved_chunks:
        source_id = chunk.get("paper_id", "")
        text = chunk.get("text", "")
        if not source_id or not text:
            continue
        for match in INLINE_CITATION_PATTERN.finditer(text):
            for piece in match.group(1).split(","):
                piece = piece.strip()
                if piece.isdigit():
                    result.setdefault(source_id, set()).add(int(piece))
    return {k: sorted(v) for k, v in result.items()}
