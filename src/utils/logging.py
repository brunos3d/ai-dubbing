"""Structured logging with stage prefixes."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from .paths import log_dir

_STAGE_NAMES = [
    "Audio Extraction",
    "Demucs Separation",
    "Pyannote Diarization",
    "Sample Extraction",
    "Whisper Transcription",
    "Translation",
    "OmniVoice Generation",
    "Duration Alignment",
    "Reconstruction",
    "Final Mix",
    "Optional Video",
]


def stage_name(idx: int) -> str:
    if 0 <= idx < len(_STAGE_NAMES):
        return _STAGE_NAMES[idx]
    return f"Stage {idx + 1}"


class StageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            record.stage = ""
        return True


def setup_logging(
    name: str = "ai-dubbing",
    level: int = logging.INFO,
    logfile: Optional[Path] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(stage)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = Console(force_terminal=True)
    rich_handler = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        rich_tracebacks=True,
        markup=False,
    )
    rich_handler.setFormatter(logging.Formatter("[%(stage)s] %(message)s"))
    rich_handler.addFilter(StageFilter())
    logger.addHandler(rich_handler)

    if logfile is None:
        logfile = log_dir() / "pipeline.log"

    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(StageFilter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "ai-dubbing") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


def stage_banner(logger: logging.Logger, idx: int, total: int, name: str) -> None:
    extra = {"stage": f"{idx + 1}/{total}"}
    logger.info(f"[{idx + 1}/{total}] {name}", extra=extra)
