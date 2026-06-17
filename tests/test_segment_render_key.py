"""Segment-level rendering tests (spec Phase 4).

Uses a fake OmniVoice model (no GPU, no network) to verify that the generate
stage:

* synthesizes every segment on a cold run, and
* on a re-run after editing ONE segment's text, re-synthesizes only the
  changed segment and reuses the rest (render-key skip).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.generate import GenerateStage  # noqa: E402


class _FakeOmni:
    sampling_rate = 24000

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs["text"])
        return [np.zeros(12000, dtype="float32")]  # 0.5 s @ 24 kHz


def _setup(tmp_path: Path):
    # A reference wav for the single speaker.
    spk_dir = tmp_path / "speaker_profiles" / "speaker_01" / "primary"
    spk_dir.mkdir(parents=True)
    ref = spk_dir / "reference.wav"
    sf.write(str(ref), np.zeros(16000, dtype="float32"), 16000)
    # The separated speech track (for non-speech preservation paths).
    speech = tmp_path / "speech.wav"
    sf.write(str(speech), np.zeros(48000, dtype="float32"), 16000)

    speaker_profiles = {
        "speaker_01": {
            "profiles": {
                "primary": {
                    "reference": str(ref),
                    "transcript_text": "reference transcript",
                    "score": 90.0,
                }
            }
        }
    }
    return speech, speaker_profiles


def _write_segments(path: Path, texts: list[str]) -> None:
    segs = [
        {"speaker": "speaker_01", "start": float(i * 3), "end": float(i * 3 + 3),
         "text": t, "source_text": f"src{i}"}
        for i, t in enumerate(texts)
    ]
    path.write_text(json.dumps(segs))


def _run(tmp_path, speech, speaker_profiles, translated):
    from src.tts.omnivoice import OmniVoiceSynthesizer

    fake = _FakeOmni()
    stage = GenerateStage(tmp_path, target_language="en")
    # Inject the fake OmniVoice model through the pluggable synthesizer seam.
    stage._make_synthesizer = lambda: OmniVoiceSynthesizer(  # type: ignore[method-assign]
        language="en", model_obj=fake
    )
    ctx = {
        "translated_path": str(translated),
        "speech_path": str(speech),
        "speaker_profiles": speaker_profiles,
        "source_language": "pt",
        "target_language": "en",
    }
    stage.run(ctx)
    return fake


def test_cold_run_synthesizes_all(tmp_path) -> None:
    speech, profiles = _setup(tmp_path)
    translated = tmp_path / "translated.json"
    _write_segments(translated, ["Hello there", "Goodbye now"])
    fake = _run(tmp_path, speech, profiles, translated)
    assert len(fake.calls) == 2


def test_edit_one_segment_resynthesizes_only_that_one(tmp_path) -> None:
    speech, profiles = _setup(tmp_path)
    translated = tmp_path / "translated.json"

    # Cold run: 2 synthesized.
    _write_segments(translated, ["Hello there", "Goodbye now"])
    fake1 = _run(tmp_path, speech, profiles, translated)
    assert len(fake1.calls) == 2

    # Edit only the second segment's text and re-run.
    _write_segments(translated, ["Hello there", "Farewell my friend"])
    fake2 = _run(tmp_path, speech, profiles, translated)
    assert fake2.calls == ["Farewell my friend"], fake2.calls

    # The manifest marks segment 1 reused, segment 2 freshly synthesized.
    manifest = json.loads(
        (tmp_path / "generated_segments" / "manifest.json").read_text()
    )
    assert manifest[0]["reused"] is True
    assert manifest[1]["reused"] is False


def test_no_changes_reuses_everything(tmp_path) -> None:
    speech, profiles = _setup(tmp_path)
    translated = tmp_path / "translated.json"
    _write_segments(translated, ["Hello there", "Goodbye now"])
    _run(tmp_path, speech, profiles, translated)

    # Identical re-run: nothing should be synthesized.
    fake2 = _run(tmp_path, speech, profiles, translated)
    assert fake2.calls == []
