"""Lock test for the 30 s synthetic WAV fixture used by workspace tests."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "short_sample.wav"


def test_fixture_exists() -> None:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"


def test_fixture_is_mono_16k_pcm16() -> None:
    data, sr = sf.read(str(FIXTURE), always_2d=True, dtype="float32")
    assert sr == 16000, f"sample rate: {sr}"
    assert data.shape[1] == 1, f"channels: {data.shape[1]}"
    assert data.shape[0] == 30 * 16000, f"samples: {data.shape[0]}"


def test_fixture_sha256_is_64_hex() -> None:
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert len(digest) == 64, f"digest length: {len(digest)}"
    assert re.fullmatch(r"[0-9a-f]{64}", digest), f"non-hex digest: {digest}"
