"""Process-wide rate limiter for arxiv.org HTTP requests.

arXiv Terms of Use (https://info.arxiv.org/help/api/tou.html):
    "make no more than one request every three seconds, and limit requests
    to a single connection at a time."

The User Manual reinforces this: "incorporate a 3 second delay in your code."

Critically, the limit applies to *all* arxiv.org endpoints collectively —
the metadata API (export.arxiv.org/api/query), the PDF server
(arxiv.org/pdf/...), RSS feeds, and OAI-PMH. Violating triggers HTTP 429
or 503 and repeated abuse can result in IP-level blocks.

Any code that hits arxiv.org MUST call ``throttle()`` immediately before
the request.
"""

import threading
import time

# arXiv ToU minimum interval between requests (seconds).
_MIN_INTERVAL = 3.0

_lock = threading.Lock()
_last_request_monotonic = 0.0


def throttle() -> None:
    """Block until ``_MIN_INTERVAL`` seconds have elapsed since the previous
    arxiv.org request, then record the current time.

    Thread-safe: the lock serializes concurrent callers, which also satisfies
    arXiv's "single connection at a time" rule even with parallel workers.

    First call in the process is instant (no sleep) because the initial
    timestamp is 0 and monotonic time is large.
    """
    global _last_request_monotonic
    with _lock:
        elapsed = time.monotonic() - _last_request_monotonic
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request_monotonic = time.monotonic()
