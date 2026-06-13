"""Command-line interface."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from .pipeline import Pipeline
from .utils.paths import project_root


def _str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-dubbing",
        description="Local-first multilingual dubbing pipeline",
    )
    p.add_argument("--input", "-i", required=True, help="Path to input media (mp4/mkv/mov/mp3/wav/flac)")
    p.add_argument("--source-language", "-s", required=True, help="Source language code (e.g. en, pt, es)")
    p.add_argument("--target-language", "-t", required=True, help="Target language code (e.g. en, pt-BR, es)")
    p.add_argument("--output-dir", "-o", default="output", help="Where to write final outputs")
    p.add_argument("--workdir", default="working", help="Where to write intermediate files")
    p.add_argument("--whisper-model", default="large-v3", help="faster-whisper model size")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for gated pyannote models")
    p.add_argument("--target-lufs", type=float, default=-16.0, help="Loudness target for final mix")
    p.add_argument("--start-from", default="extract", help="Stage to start from (for resuming)")
    p.add_argument("--only", default=None, help="Run only a single stage (debug)")

    out = p.add_argument_group("output formats (default: produce the full dubbed video)")
    out.add_argument(
        "--audio-only",
        action="store_true",
        help="Skip the video remux; emit only final_audio.wav. The audio file is always produced internally but is the only artefact kept on disk.",
    )
    out.add_argument(
        "--no-video",
        action="store_true",
        help=argparse.SUPPRESS,  # alias kept for backward compatibility
    )
    out.add_argument(
        "--emit",
        default="auto",
        choices=("auto", "audio", "video", "both"),
        help=(
            "What to deliver. 'auto' (the default) produces the full dubbed "
            "VIDEO plus the standalone audio file; 'audio' keeps only the "
            "audio; 'video' keeps only the video; 'both' is the same as 'auto'."
        ),
    )
    return p


def _resolve_emit(args, src_suffix: str) -> str:
    """Map CLI flags to a concrete emit plan.

    Returns one of: ``audio``, ``video``, ``both``.
    """
    explicit_audio = args.audio_only or args.no_video
    if explicit_audio and args.emit not in ("auto",):
        print(
            f"Conflicting flags: --audio-only and --emit={args.emit}",
            file=sys.stderr,
        )
        sys.exit(2)
    if explicit_audio:
        return "audio"
    if args.emit == "audio":
        return "audio"
    if args.emit == "video":
        return "video"
    # auto / both / video -> default behaviour: produce video (and audio)
    return "video"


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    src_suffix = Path(args.input).suffix
    emit = _resolve_emit(args, src_suffix)
    skip_video = emit == "audio"

    pipeline = Pipeline(
        input_path=str(Path(args.input).resolve()),
        source_language=args.source_language,
        target_language=args.target_language,
        workdir=Path(args.workdir).resolve(),
        output_path=Path(args.output_dir).resolve(),
        whisper_model=args.whisper_model,
        hf_token=args.hf_token,
        target_lufs=args.target_lufs,
        skip_video=skip_video,
    )
    try:
        pipeline.run(start_from=args.start_from, only=args.only)
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir).resolve()
    final_audio = out_dir / "final_audio.wav"
    final_video = out_dir / "final_video.mp4"
    print()
    print("[ok] Dubbing complete.")
    if emit in ("audio", "both") and final_audio.exists():
        print(f"     Audio : {final_audio}")
    if emit in ("video", "both") and final_video.exists():
        print(f"     Video : {final_video}")
    if emit == "audio" and final_video.exists():
        print(
            f"     (removed: {final_video})"
            if not final_video.exists()
            else f"     (keeping: {final_video} — the --audio-only flag does not delete it)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
