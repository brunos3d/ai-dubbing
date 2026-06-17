"""Reference-ASR device/model selection (Objective #5: GPU utilization).

The clone-prompt transcriber (used by the samples and generate stages to get
``ref_text`` for the reference clip) historically ran on CPU with the ``tiny``
model. That mis-hears the prompt — which directly feeds OmniVoice's voice
clone — and leaves the GPU idle. This picks a stronger model on GPU when CUDA
has headroom, and degrades gracefully to CPU/``tiny`` otherwise.
"""
from __future__ import annotations

from typing import Optional, Tuple


def resolve_ref_asr(
    cuda_available: Optional[bool] = None,
    free_gb: Optional[float] = None,
) -> Tuple[str, str, str]:
    """Return ``(model_size, device, compute_type)`` for reference ASR.

    Pure/inspectable: the live GPU state can be injected (and is, in tests).
    When ``cuda_available`` is None it is probed from torch at call time.
    """
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available and free_gb is None:
                free, _ = torch.cuda.mem_get_info()
                free_gb = free / (1024 ** 3)
        except Exception:
            cuda_available = False

    if not cuda_available:
        return "tiny", "cpu", "int8"

    fg = free_gb if free_gb is not None else 0.0
    if fg >= 3.0:
        # Plenty of headroom: 'small' is a big accuracy jump over 'tiny' for
        # short clips at negligible cost, and gives a faithful clone prompt.
        return "small", "cuda", "float16"
    if fg >= 1.5:
        return "base", "cuda", "float16"
    # GPU present but tight: stay modest to avoid OOM next to other models.
    return "base", "cuda", "int8_float16"
