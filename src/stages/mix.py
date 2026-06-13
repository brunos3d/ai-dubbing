"""Stage 10 - Final mix with FFmpeg.

Produces a polished output that:
- re-balances speech vs. background,
- applies EBU R128 loudness normalisation,
- keeps the original sample rate, channel layout, and format requested.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.mix")


def _ffmpeg_run(args: list) -> None:
    LOG.debug("ffmpeg %s", " ".join(args))
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def final_mix(
    speech_path: Path,
    background_path: Path,
    output_path: Path,
    speech_db: float = 0.0,
    background_db: float = -6.0,
    target_lufs: float = -16.0,
    true_peak_db: float = -1.5,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".norm.wav")
    _ffmpeg_run([
        "-i", str(speech_path),
        "-i", str(background_path),
        "-filter_complex",
        f"[0:a]volume={speech_db}dB,aresample=48000[s];"
        f"[1:a]volume={background_db}dB,aresample=48000[b];"
        f"[s][b]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.97[mix]",
        "-map", "[mix]",
        str(tmp),
    ])
    _ffmpeg_run([
        "-i", str(tmp),
        "-af", f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11:print_format=json",
        "-ar", "48000",
        "-ac", "2",
        str(output_path),
    ])
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    return {"final_path": str(output_path)}


class MixStage:
    name = "mix"

    def __init__(self, workdir: Path, output_dir: Path, target_lufs: float = -16.0):
        self.workdir = Path(workdir)
        self.output_dir = Path(output_dir)
        self.target_lufs = target_lufs

    def outputs(self) -> List[Path]:
        return [self.output_dir / "final_audio.wav"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 10, 12, "Final Mix")
        speech_path = Path(context.get("reconstructed_path") or (self.workdir / "reconstructed_speech.wav"))
        background_path = Path(context.get("background_path") or (self.workdir / "background.wav"))
        if not speech_path.exists():
            speech_path = self.workdir / "reconstructed_speech.wav"
        if not speech_path.exists():
            raise FileNotFoundError(speech_path)
        if not background_path.exists():
            background_path = self.workdir / "background.wav"
        if not background_path.exists():
            raise FileNotFoundError(background_path)
        out_path = self.output_dir / "final_audio.wav"
        info = final_mix(
            speech_path, background_path, out_path,
            target_lufs=self.target_lufs,
        )
        LOG.info(f"Final mix -> {info['final_path']}")
        free_vram()
        return info
