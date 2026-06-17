"""Tests for reference-ASR device/model selection (Objective #5: GPU audit).

Reference-clip transcription (the voice-clone prompt) ran on CPU with the
'tiny' model, which mis-hears the prompt and bottlenecks the pipeline. The
selector moves it to GPU with a stronger model when CUDA is available, while
staying safe (and light) on CPU-only hosts.
"""
from __future__ import annotations

from src.utils.asr import resolve_ref_asr


def test_uses_gpu_and_stronger_model_when_cuda_available():
    model, device, compute = resolve_ref_asr(cuda_available=True, free_gb=8.0)
    assert device == "cuda"
    assert compute in ("float16", "int8_float16")
    assert model != "tiny"  # a stronger clone-prompt transcriber


def test_falls_back_to_cpu_tiny_without_cuda():
    model, device, compute = resolve_ref_asr(cuda_available=False, free_gb=0.0)
    assert device == "cpu"
    assert compute == "int8"
    assert model == "tiny"


def test_low_vram_gpu_stays_modest():
    # GPU present but little headroom -> keep a small model to avoid OOM.
    model, device, compute = resolve_ref_asr(cuda_available=True, free_gb=1.0)
    assert device == "cuda"
    assert model in ("base", "small")
