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


def free_vram_aggressive() -> None:
    """Aggressive VRAM cleanup for long-running optimization loops.
    
    This is more thorough than free_vram() and should be called between
    optimization iterations to prevent VRAM accumulation. It:
    - Forces multiple garbage collection cycles
    - Synchronizes CUDA operations
    - Clears all cached memory
    - Resets memory stats
    """
    if not torch.cuda.is_available():
        gc.collect()
        return
    
    # Multiple GC cycles to catch circular references
    for _ in range(3):
        gc.collect()
    
    # Synchronize to ensure all CUDA operations are complete
    torch.cuda.synchronize()
    
    # Clear all cached memory
    torch.cuda.empty_cache()
    
    # Collect IPC memory (shared memory between processes)
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    
    # Reset memory stats to get accurate readings
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except Exception:
        pass
    
    # Final GC to clean up any remaining references
    gc.collect()


def vram_used_mb() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() // (1024 * 1024)


def vram_reserved_mb() -> Optional[int]:
    """Get reserved (cached) VRAM in MB."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_reserved() // (1024 * 1024)


def vram_total_mb() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)


def log_vram(logger) -> None:
    if not torch.cuda.is_available():
        return
    used = vram_used_mb()
    reserved = vram_reserved_mb()
    total = vram_total_mb()
    if used is not None and total is not None:
        if reserved is not None:
            logger.info(f"VRAM: {used} MB used / {reserved} MB reserved / {total} MB total")
        else:
            logger.info(f"VRAM: {used} MB / {total} MB")
