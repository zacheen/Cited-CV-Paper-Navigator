"""CLI entry point for daily HF Daily Papers ingestion.

Thin wrapper around :func:`src.ingestion.past_trend.run_past_trend`. Uses
``days_back=1, min_new_papers=None`` semantics:

- ``days_back=1`` → process exactly today.
- ``min_new_papers=None`` → no expansion mode, no minimum paper count gate;
  just run the single day and exit.
- ``stop_on_indexed`` left at the default ``False`` so a single already-
  indexed paper doesn't make us skip other papers from the same day.

Intended for cron use; idempotent re-runs are safe because
:func:`download_pdf` skips PDFs already on disk and :func:`index_chunks`
upserts by deterministic chunk id.
"""

import sys
from pathlib import Path

# Project root onto sys.path so ``src.*`` imports resolve when this file is
# run directly as a script (not as a package member).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.past_trend import run_past_trend  # noqa: E402

if __name__ == "__main__":
    run_past_trend(days_back=1, min_new_papers=None)
