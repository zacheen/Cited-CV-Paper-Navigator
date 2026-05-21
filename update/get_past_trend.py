"""CLI entry point for past-trend HF Daily CV paper ingestion.

The actual implementation lives in :mod:`src.ingestion.past_trend` so it can
be imported normally by the Streamlit background worker
(:func:`src.rag.download_state.enqueue_past_trend_download`). This file just
makes ``python update/get_past_trend.py`` keep working from the project root.
"""

import sys
from pathlib import Path

# Project root onto sys.path so ``src.*`` imports resolve when this file is
# run directly as a script (not as a package member).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.past_trend import run_past_trend  # noqa: E402

if __name__ == "__main__":
    # Past-trend CLI: walk backwards from today filling the corpus, and
    # bail the moment we hit ground we've already synced.
    run_past_trend(days_back=7, min_new_papers=3, stop_on_indexed=True)
