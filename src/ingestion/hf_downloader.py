"""Download computer vision papers from Hugging Face Daily Papers."""

import json
import time
import urllib.error
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict

from src.config import ARXIV_CATEGORY
from src.ingestion.arxiv_downloader import NAMESPACE, download_pdf
from src.ingestion.arxiv_rate_limiter import throttle as _arxiv_throttle

HF_DAILY_API = "https://huggingface.co/api/daily_papers"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

def get_hf_daily_papers(date_str: str) -> List[dict]:
    """Fetch all Hugging Face Daily Papers for a given date.
    
    Args:
        date_str: Date string in 'YYYY-MM-DD' format.
        
    Returns:
        List of paper metadata from HF API.
    """
    url = f"{HF_DAILY_API}?date={date_str}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = response.read()
            return json.loads(data)
    except Exception as e:
        print(f"Failed to fetch HF daily papers for {date_str}: {e}")
        return []

def get_arxiv_details(paper_ids: List[str], max_retries: int = 3) -> List[dict]:
    """Fetch full arXiv metadata for a list of paper IDs.

    Compliance: routes through the process-wide arxiv rate limiter so we
    never exceed 1 request / 3s across all arxiv.org calls (API + PDFs).

    On HTTP 429 / 503, retries with exponential backoff (default 30s, 60s,
    120s). If the server returned a ``Retry-After`` header, that value is
    honored instead. Even when properly throttled, a previous run's burst
    may leave our IP in arXiv's penalty window — the backoff lets it clear.

    Args:
        paper_ids: List of arXiv IDs.
        max_retries: Number of retry attempts on 429/503 (default 3).

    Returns:
        List of parsed paper dicts formatted for the pipeline. Empty on
        unrecoverable failure.
    """
    if not paper_ids:
        return []

    id_list = ",".join(paper_ids)
    url = f"{ARXIV_API_URL}?id_list={id_list}&max_results={len(paper_ids)}"

    data = None
    for attempt in range(max_retries + 1):
        _arxiv_throttle()
        try:
            with urllib.request.urlopen(url) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_retries:
                # Prefer server-supplied Retry-After if present, else exp backoff.
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = int(retry_after) if retry_after else 30 * (2 ** attempt)
                except ValueError:
                    wait = 30 * (2 ** attempt)
                print(f"  arXiv {e.code}, backing off {wait}s "
                      f"(retry {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            print(f"Failed to fetch arXiv details for ids {id_list}: {e}")
            return []
        except Exception as e:
            print(f"Failed to fetch arXiv details for ids {id_list}: {e}")
            return []

    if data is None:
        return []

    papers = []
    try:
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

            categories = [
                c.attrib.get("term")
                for c in entry.findall("atom:category", NAMESPACE)
            ]

            pdf_link = None
            for link in entry.findall("atom:link", NAMESPACE):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"]
                    break

            if pdf_link is None: # Fallback
                pdf_link = f"http://arxiv.org/pdf/{paper_id}"

            papers.append({
                "id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published,
                "categories": categories,
                "pdf_url": pdf_link,
            })

        return papers
    except Exception as e:
        print(f"Failed to parse arXiv response for ids {id_list}: {e}")
        return []

def fetch_daily_cv_papers(date_str: str, max_papers: int = 2) -> List[dict]:
    """Fetch Top CV papers from Hugging Face on a given date.
    
    Args:
        date_str: Date string in 'YYYY-MM-DD' format.
        max_papers: Maximum number of CV papers to return.
        
    Returns:
        List of paper dicts.
    """
    hf_data = get_hf_daily_papers(date_str)
    if not hf_data:
        return []
        
    paper_ids = [item["paper"]["id"] for item in hf_data if "paper" in item and "id" in item["paper"]]
    if not paper_ids:
        return []
        
    # Get official details from arxiv to check category
    arxiv_papers = get_arxiv_details(paper_ids)
    
    cv_papers = []
    for p in arxiv_papers:
        if ARXIV_CATEGORY in p["categories"]:
            cv_papers.append(p)
            if len(cv_papers) >= max_papers:
                break
                
    return cv_papers
