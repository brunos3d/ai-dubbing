"""VRAM / GPU utilities for serial model execution on 8GB cards."""
from __future__ import annotations

import gc
from typing import Optional

import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_vram() -> None:
    """Best-effort free of GPU memory between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def vram_used_mb() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() // (1024 * 1024)


def vram_total_mb() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)


def log_vram(logger) -> None:
    if not torch.cuda.is_available():
        return
    used = vram_used_mb()
    total = vram_total_mb()
    if used is not None and total is not None:
        logger.info(f"VRAM: {used} MB / {total} MB")
