"""Generate a deterministic 30 s mono 16 kHz PCM_16 WAV for workspace tests.

Signal pattern (5 s segments): 220 Hz, silence, 440 Hz, silence, 660 Hz, silence.

Output: tests/fixtures/short_sample.wav (relative to repo root).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DURATION_S = 30
SEGMENT_S = 5
FREQS_HZ = (220.0, 440.0, 660.0)
AMPLITUDE = 0.5


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    out_path = repo_root / "tests" / "fixtures" / "short_sample.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples_per_segment = SR * SEGMENT_S
    t = np.arange(samples_per_segment, dtype=np.float32) / SR
    pieces = []
    for i in range(DURATION_S // SEGMENT_S):
        if i % 2 == 0:
            freq = FREQS_HZ[i // 2]
            pieces.append(AMPLITUDE * np.sin(2.0 * np.pi * freq * t))
        else:
            pieces.append(np.zeros(samples_per_segment, dtype=np.float32))
    audio = np.concatenate(pieces).astype(np.float32)
    assert audio.shape == (SR * DURATION_S,)

    sf.write(str(out_path), audio, SR, subtype="PCM_16")
    print(f"wrote {out_path.relative_to(repo_root)} (30s @ 16000Hz mono)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
