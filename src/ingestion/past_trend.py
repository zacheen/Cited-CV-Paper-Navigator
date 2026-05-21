"""Past-trend ingestion of HF Daily CV papers.

Walks backwards from today through HF Daily Papers, downloading and indexing
CV papers into ChromaDB. Two phases:

- Phase 1: Always walk the past ``days_back`` days (default 7).
- Phase 2: If Phase 1 yielded fewer than ``min_new_papers``, keep walking
  further back (up to ``max_days_back`` total).

In both phases, encountering an already-indexed paper stops the run
immediately (incremental-sync semantics — assumes we're behind a previously
successful run and would only be re-doing work).

Callable in three contexts:
    1. CLI: ``python update/get_past_trend.py`` (thin wrapper around this module)
    2. Streamlit background thread: see ``src.rag.download_state.enqueue_past_trend_download``
    3. Tests / future automation: ``from src.ingestion.past_trend import run_past_trend``
"""

from datetime import datetime, timedelta

from src.config import PDF_DIR
from src.ingestion.arxiv_downloader import download_pdf
from src.ingestion.hf_downloader import fetch_daily_cv_papers
from src.ingestion.pdf_parser import parse_pdf
from src.processing.chunker import chunk_document
from src.processing.embedder import get_chunk_count_fast, get_collection, index_chunks


def _is_paper_indexed(collection, paper_id: str) -> bool:
    """True iff ChromaDB already has at least one chunk for paper_id."""
    try:
        result = collection.get(where={"paper_id": paper_id}, limit=1)
        return bool(result and result.get("ids"))
    except Exception:
        return False


def run_past_trend(
    days_back: int = 7,
    min_new_papers: int | None = 3,
    max_days_back: int = 30,
    stop_on_indexed: bool = False,
) -> None:
    """Fetch, download, parse and index HF CV papers.

    Two complementary controls govern when the run stops:

    - **Expansion (``min_new_papers``)**: when set to an int, Phase 1 walks
      the past ``days_back`` days unconditionally; Phase 2 then keeps walking
      further back (up to ``max_days_back``) until at least ``min_new_papers``
      new papers have been ingested. When set to ``None``, expansion is
      disabled — the run simply walks ``days_back`` days and stops.
      ``max_days_back`` is ignored in that case.

    - **Incremental sync (``stop_on_indexed``)**: when True, the run halts
      as soon as it encounters a paper that's already in ChromaDB. This is
      the right semantics for "catch up from where I left off" runs (e.g.
      the daily/weekly past-trend CLI). When False (the default), already-
      indexed papers are skipped at the day boundary via the dedup gate
      inside ``download_pdf`` / ``index_chunks`` (upsert), but the loop
      continues — safer for one-off "process exactly today's papers" cron
      runs where a single duplicate shouldn't abort the rest of the day.
    """
    today = datetime.now()
    collection = get_collection()

    total_papers = 0
    total_chunks = 0
    # Without min_new_papers we only ever do Phase 1, so the loop bound
    # collapses to days_back. ``max_days_back`` is effectively unused.
    effective_max = days_back if min_new_papers is None else max_days_back
    stop_reason = (
        f"completed days_back={days_back}"
        if min_new_papers is None
        else f"reached max_days_back={max_days_back}"
    )

    for i in range(effective_max):
        # arXiv rate limiting is handled centrally by arxiv_rate_limiter.throttle()
        # inside get_arxiv_details / download_pdf, so no manual sleep here.

        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")

        print(f"\n--- Fetching HF Daily CV Papers for {date_str} ---")
        papers = fetch_daily_cv_papers(date_str, max_papers=2)

        if not papers:
            print("  No CV papers found.")
        else:
            print(f"  Found {len(papers)} CV papers.")

        hit_indexed = False
        for paper in papers:
            paper_id = paper["id"]

            if stop_on_indexed and _is_paper_indexed(collection, paper_id):
                print(f"  Paper {paper_id} already indexed - stopping early.")
                hit_indexed = True
                break

            pdf_path = download_pdf(paper, PDF_DIR)
            if not pdf_path:
                print(f"  Failed to download {paper_id}")
                continue

            try:
                parsed = parse_pdf(pdf_path)
                text = parsed["text"]

                if len(text.strip()) < 100:
                    print(f"  Skipping {paper_id}: too little text extracted")
                    continue

                authors = ", ".join(paper.get("authors", []))

                chunks = chunk_document(
                    text,
                    paper_id=paper_id,
                    title=paper.get("title", ""),
                    arxiv_url=paper.get("pdf_url", f"https://arxiv.org/abs/{paper_id}").replace("/pdf/", "/abs/"),
                    authors=authors,
                    published=paper.get("published", ""),
                    hf_date=date_str,  # Crucial for Recent Filtering!
                    abstract=paper.get("summary", ""),
                )

                indexed = index_chunks(chunks, collection=collection)
                total_chunks += indexed
                total_papers += 1
                print(f"  Indexed {indexed} chunks for {paper_id}")

            except Exception as e:
                print(f"  Error processing {paper_id}: {e}")
                continue

        if hit_indexed:
            stop_reason = f"hit already-indexed paper after {total_papers} new papers"
            break

        # Expansion gate only activates AFTER Phase 1 (past `days_back` days)
        # completes — Phase 1 always runs to completion regardless of count.
        # Skipped entirely when min_new_papers is None (no expansion mode).
        if min_new_papers is not None:
            days_completed = i + 1
            if days_completed >= days_back and total_papers >= min_new_papers:
                stop_reason = f"reached min_new_papers={min_new_papers} after {days_completed} days"
                break

    print(f"\nPast Trends Ingestion Complete! (stopped: {stop_reason})")
    print(f"  Total papers processed: {total_papers}")
    print(f"  New chunks indexed: {total_chunks}")
    print(f"  Total chunks in collection: {get_chunk_count_fast()}")
