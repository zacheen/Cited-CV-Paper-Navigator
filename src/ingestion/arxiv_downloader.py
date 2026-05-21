"""Download computer vision papers from arXiv using the official API."""

import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

from src.config import PDF_DIR, ARXIV_CATEGORY, ARXIV_MAX_RESULTS, ARXIV_SORT_BY
from src.ingestion.arxiv_rate_limiter import throttle as _arxiv_throttle

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str = "", max_results: int = ARXIV_MAX_RESULTS,
                 category: str = ARXIV_CATEGORY,
                 sort_by: str = ARXIV_SORT_BY) -> list[dict]:
    """Search arXiv and return paper metadata.

    Args:
        query: Free-text search query (combined with category filter).
        max_results: Maximum number of results to fetch.
        category: arXiv category to filter (e.g. 'cs.CV').
        sort_by: Sort order — 'submittedDate', 'relevance', or 'lastUpdatedDate'.

    Returns:
        List of dicts with keys: id, title, summary, authors, published, pdf_url.
    """
    search_query = f"cat:{category}"
    if query:
        search_query += f" AND all:{query}"

    papers = []
    batch_size = 100

    for start in range(0, max_results, batch_size):
        params = urllib.parse.urlencode({
            "search_query": search_query,
            "start": start,
            "max_results": min(batch_size, max_results - start),
            "sortBy": sort_by,
            "sortOrder": "descending",
        })
        url = f"{ARXIV_API_URL}?{params}"

        _arxiv_throttle()
        with urllib.request.urlopen(url) as response:
            data = response.read()

        root = ET.fromstring(data)

        for entry in root.findall("atom:entry", NAMESPACE):
            paper_id = entry.find("atom:id", NAMESPACE).text.strip().split("/abs/")[-1]
            title = entry.find("atom:title", NAMESPACE).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", NAMESPACE).text.strip().replace("\n", " ")
            published = entry.find("atom:published", NAMESPACE).text.strip()

            authors = [
                a.find("atom:name", NAMESPACE).text.strip()
                for a in entry.findall("atom:author", NAMESPACE)
            ]

            pdf_link = None
            for link in entry.findall("atom:link", NAMESPACE):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"]
                    break

            if pdf_link is None:
                pdf_link = f"http://arxiv.org/pdf/{paper_id}"

            papers.append({
                "id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published,
                "pdf_url": pdf_link,
            })

        # Rate limiting is enforced by _arxiv_throttle() above; no extra sleep needed.

    return papers


def download_pdf(paper: dict, output_dir: Path = PDF_DIR) -> Path | None:
    """Download a single paper PDF.

    Args:
        paper: Dict with at least 'id' and 'pdf_url' keys.
        output_dir: Directory to save PDFs.

    Returns:
        Path to downloaded file, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = paper["id"].replace("/", "_")
    pdf_path = output_dir / f"{safe_id}.pdf"

    if pdf_path.exists():
        return pdf_path

    try:
        _arxiv_throttle()
        urllib.request.urlretrieve(paper["pdf_url"], pdf_path)
        return pdf_path
    except Exception as e:
        print(f"Failed to download {paper['id']}: {e}")
        return None


def download_papers(papers: list[dict], output_dir: Path = PDF_DIR) -> list[Path]:
    """Download a batch of papers, respecting rate limits.

    Args:
        papers: List of paper metadata dicts.
        output_dir: Directory to save PDFs.

    Returns:
        List of paths to successfully downloaded PDFs.
    """
    paths = []
    for i, paper in enumerate(papers):
        path = download_pdf(paper, output_dir)
        if path:
            paths.append(path)
        if (i + 1) % 10 == 0:
            print(f"Downloaded {i + 1}/{len(papers)} papers")
        # Respect rate limits
        time.sleep(1)
    return paths
