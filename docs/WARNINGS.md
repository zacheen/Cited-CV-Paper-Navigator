# WARNINGS — Things to Watch Out For

Implementation choices in this codebase that are correct **today** but rest
on assumptions that could quietly break later. Each entry names the
assumption, explains why we accept it, and points to the contingency plan
if the assumption fails.

This file is for **proactive warnings** ("read this before you change X").
For post-mortems of bugs we already hit, see [LESSONS.md](LESSONS.md).

---

## `get_chunk_count_fast()` uses `MAX(rowid)`, not `COUNT(*)` (2026-05-07)

**File:** [src/processing/embedder.py](../src/processing/embedder.py) — `get_chunk_count_fast()`

**Why this exists:** The Streamlit sidebar's `Indexed chunks` metric used to
call `Collection.count()` on the ChromaDB collection. On a freshly-rebooted
machine with the OS file-system cache empty, that call took **94 seconds**
(triggers segment / HNSW initialization on first cold access). Even calling
SQLite directly with `SELECT COUNT(*) FROM embeddings` took **~62 seconds**
because COUNT walks the entire B-tree (~6000 leaf pages for 303k rows on a
2.3 GB DB). Both were on the page-load critical path.

`MAX(rowid)` instead follows only the rightmost path of the rowid B-tree,
which is **O(log N)** — about 4 page reads regardless of table size,
sub-second even on cold disk.

**The assumption:** `MAX(rowid) == COUNT(*)`. This holds only when:

1. rowids are sequential starting from 1 (default SQLite `INSERT` behavior
   — ChromaDB does not override this).
2. No row has ever been deleted from the `embeddings` table.

Both are currently true: ingestion only ever calls `collection.upsert(...)`,
which inserts new chunks or updates existing ones — never deletes. The
project has no "remove paper" feature.

**When this breaks:**

- **Adding chunk-deletion functionality.** If anyone calls
  `collection.delete(...)` or implements a "remove old papers" feature,
  rowids develop holes. `MAX(rowid)` will start over-reporting.
- **ChromaDB internal changes.** A schema migration or version upgrade
  that re-creates the `embeddings` table with `AUTOINCREMENT` semantics
  would also break the equivalence (deletes never re-use rowids under
  AUTOINCREMENT).
- **Manual SQLite editing.** Any direct `INSERT (rowid, ...) VALUES (1000, ...)`
  or `DELETE` would skew the result.

**How to detect:** Compare the two values once on warm cache:
```powershell
& "$env:USERPROFILE\miniconda3\envs\ML\python.exe" -c @"
import sqlite3
from src.config import CHROMA_DIR
conn = sqlite3.connect(f'file:{CHROMA_DIR.as_posix()}/chroma.sqlite3?mode=ro', uri=True)
print(conn.execute('SELECT MAX(rowid), COUNT(*) FROM embeddings').fetchone())
"@
```
If the two numbers differ, `MAX(rowid)` is no longer safe.

**Migration plan if it breaks:** Switch to a **persistent counter file**:

1. Maintain `data/chroma_db/.chunk_count` as a single-line text file holding
   the current count.
2. Update it at the end of every ingestion script
   ([update/ingest.py](../update/ingest.py),
   [update/get_today_trend.py](../update/get_today_trend.py),
   [update/get_past_trend.py](../update/get_past_trend.py))
   by calling `Collection.count()` once on the warm path.
3. `get_chunk_count_fast()` reads the file (< 1 ms) instead of querying SQLite.

This trades real-time accuracy (count is only as fresh as the last ingest)
for guaranteed fast startup independent of SQL/disk behavior.
