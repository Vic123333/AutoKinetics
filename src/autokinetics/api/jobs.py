"""Execution of CPU-heavy work in a process pool with time limits and back-pressure.

Workers receive plain JSON-compatible payloads, rebuild the mechanism and return
plain dicts, so nothing unpicklable crosses the process boundary.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from ..errors import AutoKineticsError
from .settings import settings

log = logging.getLogger("autokinetics.jobs")


class Overloaded(Exception):
    pass


class JobTimeout(Exception):
    pass


class JobRunner:
    def __init__(self, workers: int, max_queue: int):
        self.workers = workers
        self.max_queue = max_queue
        self._pool: ProcessPoolExecutor | None = None
        self._pending = 0
        self._lock = asyncio.Lock()

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    async def run(self, fn: Callable[..., Any], *args: Any, timeout: float | None = None) -> Any:
        async with self._lock:
            if self._pending >= self.workers + self.max_queue:
                raise Overloaded
            self._pending += 1
        t0 = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(self._ensure_pool(), fn, *args)
            try:
                return await asyncio.wait_for(fut, timeout=(timeout or settings.max_wall_time) + 15.0)
            except asyncio.TimeoutError:
                # The worker keeps running until its own wall-time check fires; recycle the pool
                # if that safety net failed, so stuck workers cannot accumulate.
                log.warning("job %s exceeded its time limit; recycling worker pool", getattr(fn, "__name__", fn))
                self.shutdown()
                raise JobTimeout from None
            except BrokenProcessPool:
                self.shutdown()
                raise AutoKineticsError("Rechenprozess abgestürzt (Speicher?) – Anfrage verkleinern") from None
        finally:
            async with self._lock:
                self._pending -= 1
            log.info("job %s finished in %.2f s", getattr(fn, "__name__", fn), time.perf_counter() - t0)


runner = JobRunner(settings.workers, settings.max_queue)
