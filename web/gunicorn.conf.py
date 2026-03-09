"""
Gunicorn production configuration.
Environment variables override these defaults.
"""
import multiprocessing
import os

# ── workers ────────────────────────────────────────────────────────────────
# Simulation is CPU-bound; keep worker count modest.
workers     = int(os.getenv('GUNICORN_WORKERS', max(2, multiprocessing.cpu_count())))
worker_class = 'sync'
threads     = 1    # sync workers are single-threaded; ThreadPoolExecutor handles concurrency

# ── network ────────────────────────────────────────────────────────────────
bind        = f"0.0.0.0:{os.getenv('PORT', '8000')}"
timeout     = 60   # graceful shutdown after 60s (must exceed SIMULATION_TIMEOUT_S)
keepalive   = 5

# ── logging ────────────────────────────────────────────────────────────────
accesslog   = '-'
errorlog    = '-'
loglevel    = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── security ───────────────────────────────────────────────────────────────
limit_request_line   = 4096    # max URL length
limit_request_fields = 50      # max HTTP header fields
