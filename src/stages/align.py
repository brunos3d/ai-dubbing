"""Stage 8 - Duration alignment.

Bring each generated segment's duration close to its original duration by
re-generating with explicit `duration` if available, and otherwise
time-stretching with FFmpeg's ``atempo`` (with librosa as a fallback).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from ..utils.audio import read_wav, time_stretch, write_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.align")


def _stretch_with_ffmpeg(input_path: Path, output_path: Path, rate: float) -> bool:
    """Use ffmpeg's atempo filter for fast, high-quality time-stretching.

    atempo is limited to [0.5, 100.0]; for values outside that range, the
    caller chains multiple atempo filters together.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    if rate <= 0:
        return False
    filters: list = []
    r = rate
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r *= 2.0
    filters.append(f"atempo={r:.4f}")
    af = ",".join(filters)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-filter:a", af,
        "-ar", "24000",
        "-ac", "1",
        str(output_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return False
    if proc.returncode != 0:
        LOG.debug(f"ffmpeg atempo failed: {proc.stderr[:200]}")
    return proc.returncode == 0


def _stretch_one(input_path: Path, output_path: Path, target_duration: float) -> float:
    try:
        audio, sr = sf.read(str(input_path), always_2d=True, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        LOG.warning(f"Could not read {input_path} for stretching: {exc}; copying as-is")
        if input_path.resolve() != output_path.resolve():
            import shutil as _sh
            _sh.copy2(input_path, output_path)
        try:
            audio_fb, sr_fb = sf.read(str(input_path), always_2d=True, dtype="float32")
            return audio_fb.shape[0] / sr_fb
        except Exception:
            return target_duration
    cur = audio.shape[0] / sr
    if cur <= 0.001:
        return cur
    if abs(cur - target_duration) / max(target_duration, 0.05) < 0.02:
        if output_path.resolve() != input_path.resolve():
            write_wav(output_path, audio, sr)
        return cur
    rate = cur / max(target_duration, 0.05)
    if rate <= 0:
        if output_path.resolve() != input_path.resolve():
            write_wav(output_path, audio, sr)
        return cur
    if _stretch_with_ffmpeg(input_path, output_path, rate):
        try:
            new, _ = sf.read(str(output_path), always_2d=True, dtype="float32")
            return new.shape[0] / sr
        except Exception:
            write_wav(output_path, audio, sr)
            return cur
    try:
        stretched = time_stretch(audio, rate=rate)
        write_wav(output_path, stretched, sr)
        return stretched.shape[-1] / sr
    except Exception as exc:  # noqa: BLE001
        LOG.warning(f"time_stretch failed for {input_path}: {exc}; copying original")
        if output_path.resolve() != input_path.resolve():
            write_wav(output_path, audio, sr)
        return cur


def _regenerate_with_duration(
    model,
    segment: Dict[str, Any],
    samples_map: Dict[str, str],
    target_language: str,
    target_duration: float,
) -> Optional[np.ndarray]:
    from .generate import _coerce_ref_audio, _lang_name

    text = segment.get("text", "")
    speaker = segment.get("speaker", next(iter(samples_map)))
    if speaker not in samples_map:
        speaker = next(iter(samples_map))
    wav, sr = _coerce_ref_audio(samples_map[speaker])
    try:
        audios = model.generate(
            text=text,
            language=_lang_name(target_language),
            ref_audio=(wav, sr),
            ref_text=None,
            duration=float(target_duration),
        )
    except Exception as exc:  # noqa: BLE001
        LOG.debug(f"regenerate with duration failed: {exc}")
        return None
    if not audios:
        return None
    audio = audios[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class AlignStage:
    name = "align"
    inputs: List[str] = ["generated_segments/manifest.json"]
    outputs: List[str] = ["aligned_manifest.json"]
    editable_outputs: List[str] = []
    derived_outputs: List[str] = ["aligned_manifest.json"]
    config_fields: List[str] = [
        "target_language",
        "tolerance",
        "regenerate_with_duration",
    ]

    def __init__(
        self,
        workdir: Path,
        target_language: str,
        tolerance: float = 0.10,
        out_dir_name: str = "aligned_segments",
        regenerate_with_duration: bool = False,
        subdir: str | None = None,
    ):
        self.workdir = Path(workdir)
        if subdir:
            self.workdir = self.workdir / subdir
        self.target_language = target_language
        self.tolerance = tolerance
        self.out_dir = self.workdir / out_dir_name
        self.regenerate_with_duration = regenerate_with_duration
        self.subdir = subdir

    def output_paths(self) -> List[Path]:
        return [self.out_dir, self.workdir / "aligned_manifest.json"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 7, 11, "Duration Alignment")
        generated_dir = Path(context["generated_dir"])
        manifest_path = Path(context["manifest_path"])
        samples_map: Dict[str, str] = context.get("speaker_samples", {})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        aligned_manifest: List[Dict[str, Any]] = []
        model = None
        if self.regenerate_with_duration and samples_map:
            try:
                import os
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                from omnivoice import OmniVoice
                try:
                    free, _ = torch.cuda.mem_get_info()
                    free_gb = free / (1024 ** 3)
                except Exception:
                    free_gb = 0.0
                if free_gb < 3.5:
                    LOG.info("Loading OmniVoice with device_map='auto' for re-generation")
                    model = OmniVoice.from_pretrained(
                        "k2-fsa/OmniVoice",
                        device_map="auto",
                        max_memory={0: f"{max(0.8, free_gb - 1.7):.1f}GiB", "cpu": "30GiB"},
                        dtype=torch.float16,
                        offload_folder="/tmp/opencode/omnivoice_offload",
                    )
                else:
                    model = OmniVoice.from_pretrained(
                        "k2-fsa/OmniVoice",
                        device_map="cuda",
                        dtype=torch.float16,
                    )
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"Could not load OmniVoice for re-generation: {exc}")
                model = None

        for entry in manifest:
            src_path = Path(entry["path"])
            target_dur = float(entry.get("original_duration") or entry.get("target_duration") or 0.0)
            target_dur = max(0.1, target_dur)
            cur_dur = float(entry.get("generated_duration", 0.0))
            out_path = self.out_dir / src_path.name
            chosen_dur = cur_dur
            method = "passthrough"
            LOG.info(
                f"Aligning {src_path.name}: cur={cur_dur:.2f}s -> target={target_dur:.2f}s"
            )

            new_audio: Optional[np.ndarray] = None
            new_sr = entry.get("sample_rate", 24000)
            if (
                model is not None
                and target_dur > 0
                and abs(cur_dur - target_dur) / max(target_dur, 0.05) > self.tolerance
            ):
                seg = {"text": entry.get("text", ""), "speaker": entry.get("speaker")}
                new_audio = _regenerate_with_duration(
                    model, seg, samples_map, self.target_language, target_dur,
                )
                if new_audio is not None and new_audio.size > 0:
                    method = "regenerate_duration"
                    chosen_dur = new_audio.shape[0] / float(new_sr)

            if (
                new_audio is None
                and target_dur > 0
                and abs(cur_dur - target_dur) / max(target_dur, 0.05) > self.tolerance
            ):
                chosen_dur = _stretch_one(src_path, out_path, target_dur)
                method = "time_stretch"
            else:
                if out_path.resolve() != src_path.resolve():
                    audio, sr = read_wav(str(src_path), dtype="float32", always_2d=True)
                    write_wav(out_path, audio, sr)
            if new_audio is not None and method == "regenerate_duration":
                audio = new_audio[np.newaxis, :]
                sr = int(new_sr)
                write_wav(out_path, audio, sr)

            aligned_manifest.append({
                **entry,
                "aligned_path": str(out_path),
                "aligned_duration": round(chosen_dur, 3),
                "alignment_method": method,
                "target_duration": round(target_dur, 3),
                "duration_delta_s": round(chosen_dur - target_dur, 3),
            })

        if model is not None:
            try:
                del model
            except Exception:
                pass
            free_vram()

        manifest_out = self.workdir / "aligned_manifest.json"
        manifest_out.write_text(json.dumps(aligned_manifest, indent=2, ensure_ascii=False))
        log_vram(LOG)
        return {"aligned_dir": str(self.out_dir), "manifest_path": str(manifest_out)}
