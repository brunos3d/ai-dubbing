"""Stage 11 - Optional video remux with the new audio track."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger, stage_banner

LOG = get_logger("ai-dubbing.video")


def remux_video(
    source_video: Path,
    new_audio: Path,
    output_video: Path,
) -> Dict[str, Any]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not on PATH")
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_video),
        "-i", str(new_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_video),
    ]
    LOG.debug("ffmpeg %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg remux failed: {proc.stderr.strip()}")
    return {"video_path": str(output_video)}


class VideoStage:
    name = "video"

    def __init__(self, workdir: Path, output_dir: Path, input_path: str):
        self.workdir = Path(workdir)
        self.output_dir = Path(output_dir)
        self.input_path = Path(input_path)

    def outputs(self) -> List[Path]:
        return [self.output_dir / "final_video.mp4"]

    def run(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        stage_banner(LOG, 10, 11, "Optional Video")
        if not self.input_path.exists():
            LOG.warning("Original media missing; skipping video output")
            return None
        suffix = self.input_path.suffix.lower()
        if suffix not in {".mp4", ".mkv", ".mov"}:
            LOG.info(f"Source is {suffix} (not a video); skipping video output")
            return None
        final_audio = Path(context["final_path"])
        if not final_audio.exists():
            raise FileNotFoundError(final_audio)
        out_path = self.output_dir / "final_video.mp4"
        info = remux_video(self.input_path, final_audio, out_path)
        LOG.info(f"Video -> {info['video_path']}")
        return info
