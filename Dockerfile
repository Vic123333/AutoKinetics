# syntax=docker/dockerfile:1
# --- 1. frontend --------------------------------------------------------------
FROM node:20-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npx tsc -b && npx vite build --outDir /app/static

# --- 2. C++/Cython extension + Python wheel -------------------------------------
FROM python:3.11-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml setup.py README.md LICENSE.md ./
COPY cpp/ cpp/
COPY src/ src/
COPY --from=frontend /app/static/ src/autokinetics/static/
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# --- 3. runtime -------------------------------------------------------------------
FROM python:3.11-slim-bookworm
RUN useradd --create-home --uid 10001 autokinetics
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
USER autokinetics
ENV AUTOKINETICS_CACHE_DIR=/home/autokinetics/.cache/autokinetics \
    AUTOKINETICS_WORKERS=2 \
    AUTOKINETICS_MAX_WALL_TIME=120 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["autokinetics", "serve", "--host", "0.0.0.0", "--port", "8000"]
