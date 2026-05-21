# Design Doc — Native Function Calling Features

Status: Draft
Last updated: 2026-05-06
Owner: zacheen

## Goal

Extend the RAG chatbot with two LLM-driven features built on Gemini's
**Native Function Calling** (Automatic Function Calling, AFC):

1. **Time-range query** — let the user specify the search window in natural
   language ("papers from March 2025", "since 2024-01-01"). The LLM extracts
   the range and writes it to a session-scoped variable, which persists across
   subsequent prompts until cleared.
2. **Cited-paper download** — when retrieved chunks contain inline citations
   like `[12]`, the LLM can call a tool that resolves those indices via
   Semantic Scholar, downloads the cited papers, and ingests them into the
   ChromaDB corpus in the background.

The architecture wraps the existing retrieve → generate flow with a single
**pre-RAG function-calling pass** that handles all tool calls (intent
extraction). Retrieve always runs unconditionally as plain app code after
Stage 1, and the final generation pass streams the answer with no tools
attached. Inline `[X]` citations in retrieved chunks are handled by a
deterministic regex path that does not need an LLM round-trip.

---

## Non-goals

- Hand-rolling a ReAct loop. (Tracked separately.)
- Replacing the Ollama backend; only the Gemini path gets function calling.
- Persisting the time range across browser refreshes / sessions.
- Resolving non-arXiv references (CVPR-only, books, etc.) — those are logged
  as failures and skipped.

---

## User-visible behavior

### Sidebar changes
- **Remove**: "Only search recent 7 days" checkbox and the recent-only lock.
  ("Summarize new papers from last 7 days" button stays — it's a separate
  preset flow.)
- **Add**: read-only **"Current search range"** display showing
  `start_date → end_date` (or "All time" if cleared).
- **Add**: **"Download log"** panel — last N download attempts with
  success/failure status (e.g. `arXiv:2103.12345 — OK`, `arXiv:2104.99999 — failed`).

### Chat behavior
- User can say "show me papers from January 2025" → LLM calls
  `set_time_range("2025-01-01", "2025-01-31")` → sidebar updates → next
  retrieve uses that range.
- User says "ignore the time filter" → LLM calls `clear_time_range()` →
  sidebar shows "All time".
- If retrieved chunks contain `[12]` `[34]` etc., LLM may call
  `download_cited_papers(source_paper_id, [12, 34])` → background pipeline
  fires → current answer is generated immediately from the **already-retrieved**
  context (download does NOT block the answer). A short footer is appended:

  > *Attempted to download cited papers (`[12]`, `[34]`). If the answer above
  > is incomplete, the new papers may now be in the database — please ask
  > again.*

- If the user submits a new prompt while a download is still in flight, the
  app blocks (with a spinner) immediately before the next retrieve, until
  ingestion finishes. This guarantees the second query sees the new corpus.

---

## Architecture

### Flow per chat turn

```
user prompt
  │
  ▼
[Stage 0]  if download_in_progress: wait (spinner)
  │
  ▼
[Stage 1]  PRE-RAG function-calling pass (non-streaming)
           Gemini call with tools=[set_time_range, clear_time_range,
                                   download_cited_papers,
                                   rewrite_retrieval_query]
           contents=(user prompt + brief instruction: "extract intent —
           date filter? explicit citation download request? noisy phrasing
           that needs cleaning for retrieval? — and call the appropriate
           tools. Otherwise do nothing.")
              ├── may call set_time_range / clear_time_range
              │     → mutate st.session_state["time_range"]
              ├── may call download_cited_papers (path B, explicit only)
              │     → enqueue via shared dedup gates
              │     → fires background thread
              └── may call rewrite_retrieval_query
                    → store cleaned query in retrieval_query_state
                      (per-turn module state, cleared at start of next
                       Stage 1)
              Tool return values are discarded — Stage 1's text output is
              not shown to the user.
  │
  ▼
[Stage 2]  retrieve(query, time_range)
              query = retrieval_query_state.cleaned if set, else raw prompt
           App code, always runs, uses the just-updated range.
  │
  ▼
[Stage 2.5] regex auto-trigger (path A, no LLM)
           extract_inline_citations(retrieved_chunks) → for each
           (source_paper_id, [indices]) call enqueue_citation_download(...)
  │
  ▼
[Stage 3]  Generation pass (streaming, NO tools)
           Gemini call with contents=(raw prompt + retrieved context),
           system_instruction=RAG_SYSTEM_PROMPT
              └── streams final answer text to UI
           Note: uses the raw prompt, not the cleaned one — the user's
           phrasing carries intent that the answer should respect.
  │
  ▼
[Stage 4]  If any download was queued (path A or B), append
           "attempted download" footer.
           Sidebar shows updated range + download log on the NEXT rerun.
```

Why this shape:
- All tool calls are **intent extraction** from the user prompt — none of
  them need retrieved context. Concentrating them in Stage 1 lets retrieve
  benefit from `set_time_range` in the same turn.
- Retrieve is **app code, not a tool** — guarantees it always runs and keeps
  control flow deterministic. Letting the LLM call retrieve would be a more
  agentic / ReAct-style design, out of scope here.
- Stage 3 has no tools, so streaming behavior is the standard
  `generate_content_stream` path — no AFC interleaving concerns.
- Path A (regex auto-trigger) is fully deterministic and lives between
  Stage 2 and Stage 3. It does not require an LLM round-trip.

### Session state additions

```python
st.session_state["time_range"] = {
    "start_date": "2024-01-01",  # ISO date or None
    "end_date":   "2026-05-06",  # ISO date or None
}  # None for both = no filter

st.session_state["download_jobs"] = []
# each job: {"arxiv_id": str, "status": "pending"|"ok"|"failed",
#            "error": str|None, "started_at": iso, "finished_at": iso|None}

st.session_state["download_in_progress"] = False  # set/cleared by background thread
```

---

## Tool specifications

All three tools are plain Python functions registered via
`types.GenerateContentConfig(tools=[...])`. Gemini's AFC will introspect their
docstrings and signatures.

### 1. `set_time_range(start_date, end_date)`

```python
def set_time_range(start_date: str, end_date: str) -> str:
    """Set the publication-date range for paper retrieval.

    Call this when the user wants to filter papers by date. The LLM is
    responsible for converting natural language ("last month", "March 2025",
    "since 2024") into absolute ISO dates. Today's date is provided in the
    system prompt — use it to resolve relative phrases.

    Args:
        start_date: ISO date string YYYY-MM-DD (inclusive).
        end_date:   ISO date string YYYY-MM-DD (inclusive). Defaults to
                    today if the user did not specify an end date — the LLM
                    should pass today's date explicitly.

    Returns:
        Confirmation string, e.g. "Time range set to 2025-01-01 → 2025-01-31."
    """
```

Side effect: writes to `st.session_state["time_range"]`.

### 2. `clear_time_range()`

```python
def clear_time_range() -> str:
    """Remove the publication-date filter. Call when the user wants to
    search across all indexed papers regardless of date.

    Returns:
        Confirmation string, e.g. "Time range cleared. Searching all papers."
    """
```

### 3. `download_cited_papers(source_paper_id, citation_indices)`

```python
def download_cited_papers(source_paper_id: str,
                          citation_indices: list[int]) -> str:
    """Download papers cited by a source paper, by their inline citation
    numbers. Use this ONLY when the user explicitly asks to download
    papers cited by a specific source paper that is NOT in the retrieved
    context (e.g. user pastes an arXiv id and asks for its references).
    For citations like [12] visible in the retrieved context, the app
    auto-triggers downloads via a deterministic regex path — do not call
    this tool in that case to avoid duplicate work.

    Args:
        source_paper_id: arXiv ID of the paper containing the citations.
        citation_indices: List of citation numbers as they appear in the
                          source paper's text (e.g. [12, 34]).

    Returns:
        Status string summarizing what was queued.
        Actual download/ingest happens asynchronously; check the sidebar
        download log for results.
    """
```

Side effect: hands the request to the shared `enqueue_citation_download(...)`
helper described below.

### Two trigger paths for cited-paper download

Both paths share the same backend (`enqueue_citation_download`) and the same
dedup queue.

| Path | Trigger | When it fires |
|---|---|---|
| **A. Regex auto-trigger** | App scans every retrieved chunk for `\[\d+\]` patterns after Stage 2 retrieve | Always, deterministically, on every chat turn (Q15 "全下") |
| **B. LLM tool call** | Gemini calls `download_cited_papers(...)` during Stage 3 | Only when user explicitly asks to download citations of a paper not in retrieved context |

The tool docstring explicitly tells the LLM not to duplicate work that path A
already covers.

### Shared dedup + DB check (`enqueue_citation_download`)

Before *any* arxiv id is queued for download (whether from path A or path B),
the helper runs these gates **in order**:

1. **ChromaDB check** — query the collection's metadata for any chunk with
   `paper_id == arxiv_id`. If present, skip (already indexed). Log as
   `"already in DB"`.
2. **PDF file check** — `(PDF_DIR / f"{safe_id}.pdf").exists()`. If present
   but step 1 missed it, schedule re-ingest only (no re-download). Log as
   `"re-indexed"`.
3. **In-flight job check** — query `src/rag/download_state.py`'s job queue.
   If a job for this arxiv id is `pending` or `running`, skip. Log nothing
   (silent dedup).
4. Otherwise: enqueue a new job. Mark `download_in_progress = True`.

Only after these checks does the helper hit Semantic Scholar / download the
PDF. This keeps the DB check authoritative — the same paper never gets
processed twice across the lifetime of the corpus.

### Background pipeline (per queued job)

1. If arxiv_id was supplied directly (path B with explicit ids), skip to 3.
2. Otherwise call Semantic Scholar:
   `GET https://api.semanticscholar.org/graph/v1/paper/arXiv:{source_paper_id}/references?fields=externalIds,title`
   and map `citation_indices[i] → references[i-1].externalIds.ArXiv` (1-indexed).
   Refs without an arXiv id are logged as `"no arXiv id"` failures and skipped.
3. For each remaining arxiv id, **re-run the dedup gates** (the corpus may
   have changed since the job was queued).
4. Run `download_pdf` → `parse_pdf` → `chunk_document` → `index_chunks`
   (same pipeline as `update/ingest.py`).
5. Update `download_state` job entry. Clear `download_in_progress` when the
   queue empties.

---

## Implementation plan

### Files to add
- `src/rag/tools.py` — the three tool functions, plus a `get_tools()` helper
  that returns the list to pass to `GenerateContentConfig`.
- `src/rag/download_state.py` — module-level job queue + dedup helpers
  (`is_in_db(arxiv_id)`, `is_in_flight(arxiv_id)`, `enqueue_citation_download(...)`,
  `extract_inline_citations(retrieved_chunks) -> dict[paper_id, list[int]]`).
- `src/ingestion/citation_resolver.py` — Semantic Scholar wrapper:
  `resolve_references(arxiv_id) -> list[dict]`.
- `src/ingestion/ingest_single.py` — refactor of `update/ingest.py`'s inner
  loop into `ingest_paper(arxiv_id_or_paper_dict)` so it's reusable from a
  background thread.

### Files to modify
- `src/rag/generator_gemini.py` — add one new entry point:
  - `run_pre_rag_pass(prompt) -> None` — non-streaming Gemini call with
    `tools=[set_time_range, clear_time_range, download_cited_papers]`.
    Side-effects only; return value ignored.

  Keep existing `generate_answer_stream` for Stage 3 (it already does
  streaming with no tools — exactly what we need).
- `src/rag/retriever.py` — replace `recent_days: int | None` parameter on
  `retrieve()` with `start_date: date | None, end_date: date | None`. Update
  the metadata-date filter accordingly.
- `app.py`:
  - Remove "Only search recent 7 days" checkbox + lock UI.
  - Add "Current search range" display + "Download log" panel.
  - Initialize the new session state keys.
  - In `run_query`, before calling retrieve, check `download_in_progress` and
    block with `st.spinner` until cleared.
  - Before Stage 2 retrieve, call `run_pre_rag_pass(prompt)` so any
    `set_time_range` / `download_cited_papers` tool calls take effect.
  - After Stage 2 retrieve, call `extract_inline_citations(retrieved)` and
    feed each `(paper_id, indices)` pair to `enqueue_citation_download` (path A).
  - Stage 3 keeps using the existing `generate_answer_stream` (no tools).

### Files unchanged
- Ollama generator, ingestion scripts, embedder, chunker.

---

## Streamlit + threading concerns

Streamlit re-runs the script top-to-bottom on every interaction. Background
threads can:

- Safely write to ChromaDB (the local persistent client is process-safe for
  our single-user dev case).
- **NOT** safely call `st.*` APIs from outside the script thread.

Workaround: the background thread writes to a thread-safe queue / module-level
`dict` (e.g. `src/rag/download_state.py`), and the Streamlit script reads it on
each rerun to populate the sidebar log. After kicking off a download, the
script calls `st.rerun()` once the user submits the next prompt or after a
short polling interval.

Decided: download log updates **on next prompt only**. No live polling.

---

## Edge cases

| Case | Handling |
|---|---|
| LLM passes `start_date > end_date` | Tool returns error string; LLM apologizes / retries. |
| LLM passes malformed ISO date | Tool returns error string. |
| Semantic Scholar API rate-limited / down | Background job marks all indices as failed with reason. |
| Citation index out of range (e.g. `[99]` but paper only has 50 refs) | Per-index failure logged; others still process. |
| Same arXiv id requested twice (any combination of path A / B) | Dedup gates in `enqueue_citation_download`: ChromaDB metadata → PDF file → in-flight queue. Same paper never processed twice. |
| User asks about time range but no current filter active | LLM should describe "All time" — covered by including the current range in the system prompt. |
| Source paper not in DB (LLM hallucinates an ID) | Tool validates `source_paper_id` against ChromaDB metadata; rejects if absent. |
| Download still running when user closes browser | Thread is daemon; dies with the process. Already-ingested papers persist. |

---

## Resolved decisions

- All four tools (`set_time_range`, `clear_time_range`,
  `download_cited_papers`, `rewrite_retrieval_query`) live in a single
  **pre-RAG** function-calling pass
  (Stage 1). Retrieve always runs as plain app code in Stage 2. Stage 3
  is streaming generation with no tools.
- Sidebar download log updates on next prompt only (no live polling).
- Download log keeps the most recent **10** entries (FIFO eviction).
- Two trigger paths for cited-paper download:
  - Path A (regex auto): primary, fires every turn for every `[X]` in
    retrieved context. No LLM round-trip.
  - Path B (LLM tool): only for explicit user requests about papers not
    in the retrieved context.
  Both share `enqueue_citation_download` and its dedup gates (ChromaDB
  metadata → PDF file → in-flight queue).

---

## Out-of-scope follow-ups (separate task)

- Hand-rolled ReAct version for comparison (mentioned by user).
- Caching Semantic Scholar responses to avoid repeat API calls.
- Showing which retrieved chunks were fetched *because of* a previous
  cited-paper download.
