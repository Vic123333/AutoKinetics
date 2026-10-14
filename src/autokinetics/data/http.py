"""Shared HTTP client with on-disk cache, rate limiting and retries."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx

from .. import __version__
from ..errors import ExternalDataError

USER_AGENT = f"AutoKinetics/{__version__} (chemical kinetics simulator; +https://github.com/Vic123333/AutoKinetics)"
CACHE_TTL_S = float(os.environ.get("AUTOKINETICS_CACHE_TTL", 30 * 24 * 3600))
OFFLINE = os.environ.get("AUTOKINETICS_OFFLINE", "0") == "1"


def cache_dir() -> Path:
    d = Path(os.environ.get("AUTOKINETICS_CACHE_DIR", Path.home() / ".cache" / "autokinetics"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class _RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_time - now
            self.next_time = max(now, self.next_time) + self.interval
        if delay > 0:
            time.sleep(delay)


_LIMITERS = {"pubchem.ncbi.nlm.nih.gov": _RateLimiter(4.0), "webbook.nist.gov": _RateLimiter(1.0)}
_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0),
                                   headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        return _client


def fetch(url: str, *, kind: str = "text", source: str = "", ttl: float = CACHE_TTL_S,
          not_found_ok: bool = False) -> str | None:
    """GET ``url`` (cached). Returns the body, or None for 404 if ``not_found_ok``."""
    key = hashlib.sha256(url.encode()).hexdigest()
    path = cache_dir() / f"{key}.json"
    if path.exists():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - entry["time"] < ttl:
                return entry["body"]
        except (OSError, ValueError, KeyError):
            pass
    if OFFLINE:
        raise ExternalDataError(f"{source or url}: Offline-Modus und kein Cache-Eintrag vorhanden")
    host = httpx.URL(url).host
    limiter = _LIMITERS.get(host)
    last_error = ""
    for attempt in range(3):
        if limiter:
            limiter.wait()
        try:
            resp = _get_client().get(url)
        except httpx.HTTPError as e:
            last_error = f"Netzwerkfehler: {e.__class__.__name__}"
            time.sleep(0.5 * (attempt + 1))
            continue
        if resp.status_code == 404:
            if not_found_ok:
                body = None
                path.write_text(json.dumps({"time": time.time(), "body": body}), encoding="utf-8")
                return None
            raise ExternalDataError(f"{source}: nicht gefunden")
        if resp.status_code in (429, 500, 502, 503, 504):
            last_error = f"HTTP {resp.status_code}"
            time.sleep(1.0 * (attempt + 1))
            continue
        if resp.status_code >= 400:
            raise ExternalDataError(f"{source}: HTTP {resp.status_code}")
        body = resp.text
        try:
            path.write_text(json.dumps({"time": time.time(), "body": body}), encoding="utf-8")
        except OSError:
            pass
        return body
    raise ExternalDataError(f"{source or host} nicht erreichbar ({last_error})")
