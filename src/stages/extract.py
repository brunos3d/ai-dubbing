"""Stage 1 - Audio extraction with FFmpeg."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.extract")


def _run_ffmpeg(args: List[str]) -> None:
    LOG.debug("ffmpeg %s", " ".join(args))
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def _ffprobe(path: str | Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def extract_audio(input_path: str, output_wav: str | Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16-bit PCM audio at the requested sample rate."""
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_wav),
    ])
    return output_wav


class ExtractStage:
    name = "extract"

    def __init__(self, workdir: Path, sample_rate: int = 16000):
        self.workdir = Path(workdir)
        self.sample_rate = sample_rate

    def outputs(self) -> List[Path]:
        return [self.workdir / "original_audio.wav", self.workdir / "media_info.json"]

    def run(self, input_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 0, 11, "Audio Extraction")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is not on PATH")
        audio_path = self.workdir / "original_audio.wav"
        info_path = self.workdir / "media_info.json"

        extract_audio(input_path, audio_path, sample_rate=self.sample_rate)
        info = _ffprobe(input_path)
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))

        duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
        LOG.info(f"Extracted audio -> {audio_path} ({duration:.2f}s @ {self.sample_rate}Hz mono)")
        log_vram(LOG)
        free_vram()
        return {
            "audio_path": str(audio_path),
            "media_info": info,
            "duration_s": duration,
            "sample_rate": self.sample_rate,
        }
