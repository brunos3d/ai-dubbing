"""Stage 2 - Source separation with Demucs."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.audio import peak_normalize, read_wav, write_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.separate")


def _separate_demucs_api(audio_path: Path, out_dir: Path, model: str, device: str) -> Dict[str, Path]:
    import torch

    from demucs.pretrained import get_model
    from demucs.separate import apply_model, load_track, save_audio

    LOG.info(f"Loading Demucs {model} ...")
    demucs_model = get_model(model)
    demucs_model.to(device)
    demucs_model.eval()

    wav = load_track(str(audio_path), audio_channels=demucs_model.audio_channels, samplerate=demucs_model.samplerate)
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav
    ref = wav_np.mean(0)
    wav_np = (wav_np - ref.mean()) / ref.std()
    mix = torch.from_numpy(wav_np).to(device)

    LOG.info("Running Demucs inference ...")
    with torch.no_grad():
        sources = apply_model(
            demucs_model,
            mix[None],
            shifts=1,
            num_workers=0,
            progress=False,
            device=device,
            segment=7.0,
        )[0]

    sources = sources * ref.std() + ref.mean()
    sources_list = sources.cpu().numpy()
    sr = demucs_model.samplerate
    out_dir.mkdir(parents=True, exist_ok=True)
    name_map = {"vocals": "speech.wav", "other": "background.wav", "drums": "drums.wav", "bass": "bass.wav"}
    out: Dict[str, Path] = {}
    for k, name in enumerate(demucs_model.sources):
        target = out_dir / name_map.get(name, f"{name}.wav")
        audio = sources_list[k]
        save_audio(
            torch.from_numpy(audio),
            str(target),
            samplerate=sr,
            clip="rescale",
            as_float=False,
            bits_per_sample=16,
        )
        out[name] = target
    return out


def _separate_demucs_cli(audio_path: Path, out_dir: Path, model: str, device: str) -> Dict[str, Path]:
    cmd = [
        "demucs",
        "-n", model,
        "--out", str(out_dir),
        "--device", device,
        "--segment", "7",
        "--overlap", "0.1",
        "--jobs", "1",
        "--two-stems", "vocals",
        str(audio_path),
    ]
    LOG.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"demucs failed: {proc.stderr[:2000]}")
    base = out_dir / model / audio_path.stem
    return {
        "vocals": base / "vocals.wav",
        "background": base / "no_vocals.wav",
    }


class SeparateStage:
    """Run Demucs to separate vocals from background."""

    name = "separate"

    def __init__(
        self,
        workdir: Path,
        model: str = "htdemucs",
        device: str = "cuda",
        out_dir_name: str = "separation",
    ):
        self.workdir = Path(workdir)
        self.model = model
        self.device = device
        self.out_dir = self.workdir / out_dir_name

    def outputs(self) -> List[Path]:
        return [self.workdir / "speech.wav", self.workdir / "background.wav"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 1, 11, "Demucs Separation")
        audio_path = Path(context["audio_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"audio not found: {audio_path}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = _separate_demucs_api(audio_path, self.out_dir, self.model, self.device)
        except Exception as exc:
            LOG.warning(f"demucs.api failed ({exc}); falling back to CLI")
            free_vram()
            if not shutil.which("demucs"):
                raise RuntimeError("demucs CLI is not on PATH") from exc
            results = _separate_demucs_cli(audio_path, self.out_dir, self.model, self.device)

        speech_src = results.get("vocals")
        bg_src = results.get("background") or results.get("other")
        if speech_src is None or not Path(speech_src).exists():
            raise RuntimeError("Demucs did not produce a vocals track")

        speech_dst = self.workdir / "speech.wav"
        background_dst = self.workdir / "background.wav"

        if bg_src is None or not Path(bg_src).exists():
            LOG.warning("No background stem; synthesising silence background from full mix")
            full, sr = read_wav(audio_path)
            speech, _ = read_wav(speech_src)
            mono_full = full.mean(axis=0)
            mono_speech = speech.mean(axis=0)
            n = min(mono_full.shape[0], mono_speech.shape[0])
            background = mono_full[:n] - mono_speech[:n]
            background = peak_normalize(background, target_db=-12.0)
            write_wav(background_dst, background, sr)
        else:
            shutil.copy2(bg_src, background_dst)

        shutil.copy2(speech_src, speech_dst)

        LOG.info(f"Speech -> {speech_dst}")
        LOG.info(f"Background -> {background_dst}")
        log_vram(LOG)
        free_vram()
        return {
            "speech_path": str(speech_dst),
            "background_path": str(background_dst),
            "model": self.model,
        }
