"""Ingestion pipeline: download arXiv CV papers, parse, chunk, and index."""

import sys
import argparse
from pathlib import Path

# Allow running as `python update/ingest.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ARXIV_MAX_RESULTS, PDF_DIR
from src.ingestion.arxiv_downloader import search_arxiv, download_papers
from src.ingestion.pdf_parser import parse_pdf
from src.processing.chunker import chunk_document
from src.processing.embedder import get_collection, index_chunks, get_chunk_count_fast


def run_ingestion(query: str = "", max_papers: int = ARXIV_MAX_RESULTS,
                  skip_download: bool = False):
    """Run the full ingestion pipeline.

    Args:
        query: Optional search query to filter papers.
        max_papers: Maximum number of papers to download.
        skip_download: If True, only process already-downloaded PDFs.
    """
    collection = get_collection()

    # Step 1: Search and download
    if not skip_download:
        print(f"Searching arXiv for cs.CV papers (query='{query}', max={max_papers})...")
        papers = search_arxiv(query=query, max_results=max_papers)
        print(f"Found {len(papers)} papers. Downloading PDFs...")
        pdf_paths = download_papers(papers)
        print(f"Downloaded {len(pdf_paths)} PDFs to {PDF_DIR}")
        # Save metadata alongside for later use
        paper_lookup = {p["id"].replace("/", "_"): p for p in papers}
    else:
        print(f"Skipping download. Processing existing PDFs in {PDF_DIR}...")
        pdf_paths = list(PDF_DIR.glob("*.pdf"))
        paper_lookup = {}
        print(f"Found {len(pdf_paths)} existing PDFs")

    # Step 2: Parse, chunk, and index
    total_chunks = 0
    skipped_existing = 0
    for i, pdf_path in enumerate(pdf_paths):
        paper_id = pdf_path.stem

        # Skip if this paper already has chunks in the DB. Without this, a
        # re-run pays the parse + chunk + embed cost again and only avoids
        # duplicate rows because index_chunks uses upsert by deterministic id.
        existing = collection.get(where={"paper_id": paper_id}, limit=1)
        if existing and existing.get("ids"):
            skipped_existing += 1
            continue

        meta = paper_lookup.get(paper_id, {})
        title = meta.get("title", paper_id)
        # Build arXiv abstract URL from paper ID
        arxiv_id = paper_id.replace("_", "/")
        arxiv_url = meta.get("pdf_url", f"https://arxiv.org/abs/{arxiv_id}").replace("/pdf/", "/abs/")
        authors = ", ".join(meta.get("authors", []))
        published = meta.get("published", "")
        abstract = meta.get("summary", "")

        try:
            parsed = parse_pdf(pdf_path)
            text = parsed["text"]

            if len(text.strip()) < 100:
                print(f"  Skipping {paper_id}: too little text extracted")
                continue

            chunks = chunk_document(
                text, paper_id=paper_id, title=title,
                arxiv_url=arxiv_url, authors=authors, published=published,
                abstract=abstract,
            )
            indexed = index_chunks(chunks, collection=collection)
            total_chunks += indexed

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(pdf_paths)} papers ({total_chunks} chunks so far)")

        except Exception as e:
            print(f"  Error processing {paper_id}: {e}")
            continue

    # Step 3: Report
    print(f"\nIngestion complete!")
    print(f"  Papers processed: {len(pdf_paths) - skipped_existing}")
    print(f"  Papers skipped (already in DB): {skipped_existing}")
    print(f"  New chunks indexed: {total_chunks}")
    print(f"  Total chunks in collection: {get_chunk_count_fast()}")


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv CV papers into ChromaDB")
    parser.add_argument("--query", type=str, default="",
                        help="Search query to filter papers (e.g. 'object detection')")
    parser.add_argument("--max-papers", type=int, default=ARXIV_MAX_RESULTS,
                        help=f"Maximum papers to download (default: {ARXIV_MAX_RESULTS})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only process existing PDFs")
    args = parser.parse_args()

    run_ingestion(query=args.query, max_papers=args.max_papers,
                  skip_download=args.skip_download)


if __name__ == "__main__":
    main()
