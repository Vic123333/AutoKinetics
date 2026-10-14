"""Runtime configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    workers: int = field(default_factory=lambda: _int("AUTOKINETICS_WORKERS", max(1, (os.cpu_count() or 2) // 2)))
    max_queue: int = field(default_factory=lambda: _int("AUTOKINETICS_MAX_QUEUE", 16))
    max_wall_time: float = field(default_factory=lambda: _float("AUTOKINETICS_MAX_WALL_TIME", 120.0))
    max_species: int = field(default_factory=lambda: _int("AUTOKINETICS_MAX_SPECIES", 2000))
    max_reactions: int = field(default_factory=lambda: _int("AUTOKINETICS_MAX_REACTIONS", 20000))
    max_upload_bytes: int = field(default_factory=lambda: _int("AUTOKINETICS_MAX_UPLOAD", 8 * 1024 * 1024))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("AUTOKINETICS_CORS_ORIGINS", "").split(",") if o.strip()))
    static_dir: Path | None = field(default_factory=lambda: (
        Path(os.environ["AUTOKINETICS_STATIC_DIR"]) if os.environ.get("AUTOKINETICS_STATIC_DIR")
        else Path(__file__).resolve().parents[1] / "static"))
    allow_online_data: bool = field(default_factory=lambda: os.environ.get("AUTOKINETICS_OFFLINE", "0") != "1")


settings = Settings()
