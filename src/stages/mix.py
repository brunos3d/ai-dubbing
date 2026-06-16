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


def build_mix_filter(
    speech_db: float = 0.0,
    background_db: float = -6.0,
    *,
    duck: bool = True,
    reverb_wet: float = 0.0,
) -> str:
    """Construct the ``-filter_complex`` graph for the final mix (spec §6).

    * ``duck`` — when True, the background is side-chain compressed by the
      dialogue envelope (``sidechaincompress``): music/ambience drops *only*
      while someone speaks and returns in the gaps, following the NEW
      dialogue timing. This is the primary "sit in the scene" win. When
      False, the previous static ``amix`` is produced (the fallback graph).
    * ``reverb_wet`` — when > 0, a light early-reflection ``aecho`` is applied
      to the (dry, studio-clean) synthetic dialogue so it gains room space
      comparable to the original. Kept subtle and capped by the caller.
    """
    parts: list[str] = []
    sp = f"[0:a]aresample=48000,volume={speech_db}dB"
    if reverb_wet and reverb_wet > 0.01:
        out_gain = round(min(0.9, reverb_wet), 3)
        decay = round(min(0.9, reverb_wet * 0.6), 3)
        # single subtle early reflection at ~50 ms
        sp += f",aecho=0.85:{out_gain}:50:{decay}"

    if duck:
        parts.append(sp + "[sp]")
        parts.append("[sp]asplit=2[sk][s]")
        parts.append(f"[1:a]aresample=48000,volume={background_db}dB[bg]")
        parts.append(
            "[bg][sk]sidechaincompress="
            "threshold=0.02:ratio=8:attack=15:release=300[bgduck]"
        )
        parts.append(
            "[s][bgduck]amix=inputs=2:duration=longest:normalize=0,"
            "alimiter=limit=0.97[mix]"
        )
    else:
        parts.append(sp + "[s]")
        parts.append(f"[1:a]aresample=48000,volume={background_db}dB[b]")
        parts.append(
            "[s][b]amix=inputs=2:duration=longest:normalize=0,"
            "alimiter=limit=0.97[mix]"
        )
    return ";".join(parts)


def estimate_reverb(path: Path) -> float:
    """Crude reverberance estimate in ``[0, 1]`` from a speech stem.

    Reverberant rooms leave energy *trailing* after speech offsets; dry
    studio audio drops to silence quickly. We measure the median ratio of
    energy ~150 ms after a voiced→unvoiced transition to the energy at the
    transition. Returns 0.0 on any failure (safe: no reverb added).
    """
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype="float64")
        if data.size < sr // 2:
            return 0.0
        win = max(1, int(0.025 * sr))
        hop = max(1, int(0.010 * sr))
        n = 1 + (data.shape[0] - win) // hop
        if n < 20:
            return 0.0
        rms = np.array([
            np.sqrt(np.mean(data[i * hop:i * hop + win] ** 2)) for i in range(n)
        ])
        peak = float(rms.max())
        if peak <= 1e-6:
            return 0.0
        voiced = rms > 0.2 * peak
        lookahead = max(1, int(0.150 / 0.010))  # ~150 ms in frames
        ratios = []
        for i in range(1, n - lookahead):
            if voiced[i] and not voiced[i + 1]:  # offset
                at = rms[i]
                after = rms[i + lookahead]
                if at > 1e-6:
                    ratios.append(min(1.0, after / at))
        if not ratios:
            return 0.0
        return float(np.clip(np.median(ratios), 0.0, 1.0))
    except Exception:  # noqa: BLE001 - estimation is best-effort
        return 0.0


def final_mix(
    speech_path: Path,
    background_path: Path,
    output_path: Path,
    speech_db: float = 0.0,
    background_db: float = -6.0,
    target_lufs: float = -16.0,
    true_peak_db: float = -1.5,
    duck: bool = True,
    reverb_wet: float = 0.0,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".norm.wav")

    used_duck = duck
    fc = build_mix_filter(speech_db, background_db, duck=duck, reverb_wet=reverb_wet)
    try:
        _ffmpeg_run(["-i", str(speech_path), "-i", str(background_path),
                     "-filter_complex", fc, "-map", "[mix]", str(tmp)])
    except RuntimeError as exc:
        if duck:
            LOG.warning(
                "ducking/room-match filtergraph failed (%s); falling back to "
                "static mix", exc,
            )
            used_duck = False
            fc = build_mix_filter(speech_db, background_db, duck=False, reverb_wet=0.0)
            _ffmpeg_run(["-i", str(speech_path), "-i", str(background_path),
                         "-filter_complex", fc, "-map", "[mix]", str(tmp)])
        else:
            raise

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
    return {
        "final_path": str(output_path),
        "ducking": used_duck,
        "reverb_wet": round(reverb_wet, 3),
    }


class MixStage:
    name = "mix"
    inputs: List[str] = [
        "output/reconstructed_speech.wav",
        "media/background.wav",
    ]
    outputs: List[str] = ["output/final_audio.wav"]
    editable_outputs: List[str] = []
    derived_outputs: List[str] = ["output/final_audio.wav"]
    config_fields: List[str] = [
        "target_lufs", "speech_db", "background_db", "ducking", "room_match",
    ]

    def __init__(
        self,
        workdir: Path,
        output_dir: Path,
        target_lufs: float = -16.0,
        ducking: bool = True,
        room_match: bool = True,
        subdir: str | None = None,
    ):
        self.workdir = Path(workdir)
        if subdir:
            self.workdir = self.workdir / subdir
        self.output_dir = Path(output_dir)
        self.target_lufs = target_lufs
        self.ducking = ducking
        self.room_match = room_match
        self.subdir = subdir

    def output_paths(self) -> List[Path]:
        return [self.output_dir / "final_audio.wav"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 9, 11, "Final Mix")
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

        # Room match: estimate reverberance from the ORIGINAL separated speech
        # stem (not the dry synthetic one) and apply a light, capped reverb so
        # the new voice gains comparable space. Gated so we never double the
        # reverb on already-wet dialogue.
        reverb_wet = 0.0
        if self.room_match:
            orig_speech = context.get("speech_path")
            if orig_speech and Path(orig_speech).exists():
                reverberance = estimate_reverb(Path(orig_speech))
                reverb_wet = round(min(0.25, max(0.0, reverberance * 0.5)), 3)
                LOG.info(
                    f"Room match: reverberance={reverberance:.3f} -> wet={reverb_wet}"
                )

        out_path = self.output_dir / "final_audio.wav"
        info = final_mix(
            speech_path, background_path, out_path,
            target_lufs=self.target_lufs,
            duck=self.ducking,
            reverb_wet=reverb_wet,
        )
        LOG.info(
            f"Final mix -> {info['final_path']} "
            f"(ducking={info.get('ducking')}, reverb_wet={info.get('reverb_wet')})"
        )
        free_vram()
        return info
