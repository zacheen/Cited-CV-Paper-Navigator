"""Entry point to ingest past year of HF Daily Papers."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running as `python data/get_past_trend.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.hf_downloader import fetch_daily_cv_papers
from src.ingestion.arxiv_downloader import download_pdf
from src.ingestion.pdf_parser import parse_pdf
from src.processing.chunker import chunk_document
from src.processing.embedder import get_collection, index_chunks, get_chunk_count_fast
from src.config import PDF_DIR

def _is_paper_indexed(collection, paper_id: str) -> bool:
    """True iff ChromaDB already has at least one chunk for paper_id."""
    try:
        result = collection.get(where={"paper_id": paper_id}, limit=1)
        return bool(result and result.get("ids"))
    except Exception:
        return False


def run_past_trend(days_back: int = 7, min_new_papers: int = 3, max_days_back: int = 30):
    """Fetch, download, parse and index HF CV papers.

    Phase 1: Always walk the past ``days_back`` days.
    Phase 2: If Phase 1 yielded fewer than ``min_new_papers`` new papers, keep
             walking further back (up to ``max_days_back`` total) until
             ``min_new_papers`` is reached.

    In BOTH phases, if a paper is encountered that is already indexed in
    ChromaDB, the run stops immediately (incremental-sync semantics).
    """
    today = datetime.now()
    collection = get_collection()

    total_papers = 0
    total_chunks = 0
    stop_reason = f"reached max_days_back={max_days_back}"

    for i in range(max_days_back):
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

            if _is_paper_indexed(collection, paper_id):
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

        # min_new_papers gate only activates AFTER Phase 1 (past `days_back` days)
        # finishes — Phase 1 always runs to completion regardless of count.
        days_completed = i + 1
        if days_completed >= days_back and total_papers >= min_new_papers:
            stop_reason = f"reached min_new_papers={min_new_papers} after {days_completed} days"
            break

    print(f"\nPast Trends Ingestion Complete! (stopped: {stop_reason})")
    print(f"  Total papers processed: {total_papers}")
    print(f"  New chunks indexed: {total_chunks}")
    print(f"  Total chunks in collection: {get_chunk_count_fast()}")

if __name__ == "__main__":
    run_past_trend(days_back=7, min_new_papers=3)
