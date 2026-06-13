"""Command-line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .pipeline import Pipeline
from .utils.cache import CacheManager
from .utils.paths import project_root


def _str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-dubbing",
        description="Local-first multilingual dubbing pipeline",
    )
    subparsers = p.add_subparsers(dest="command")

    # Pipeline Run (default command)
    run_p = subparsers.add_parser("run", help="Run the dubbing pipeline")
    run_p.add_argument("--input", "-i", required=True, help="Path to input media (mp4/mkv/mov/mp3/wav/flac)")
    run_p.add_argument("--source-language", "-s", required=True, help="Source language code (e.g. en, pt, es)")
    run_p.add_argument("--target-language", "-t", required=True, help="Target language code (e.g. en, pt-BR, es)")
    run_p.add_argument("--output-dir", "-o", default="output", help="Where to write final outputs")
    run_p.add_argument("--workdir", help="Where to write intermediate files (overrides automatic cache path)")
    run_p.add_argument("--whisper-model", default="large-v3", help="faster-whisper model size")
    run_p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for gated pyannote models")
    run_p.add_argument("--target-lufs", type=float, default=-16.0, help="Loudness target for final mix")
    run_p.add_argument("--glossary", help="Path to entity_glossary.json for preserving names/brands")
    run_p.add_argument("--adapt-model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct", help="Model for speech adaptation")
    run_p.add_argument("--adapt-mode", default="YouTube Narrator", help="Adaptation persona (e.g. YouTube Narrator, Documentary, Casual)")
    run_p.add_argument("--start-from", default="extract", help="Stage to start from (for resuming)")
    run_p.add_argument("--only", default=None, help="Run only a single stage (debug)")
    
    # Cache Control Flags
    run_p.add_argument("--no-cache", action="store_true", help="Force rebuild from scratch")
    run_p.add_argument("--from-stage", help="Rebuild everything from the specified stage onwards")
    run_p.add_argument("--read-only-cache", action="store_true", help="Do not write new data to the cache")

    out = run_p.add_argument_group("output formats (default: produce the full dubbed video)")
    out.add_argument(
        "--audio-only",
        action="store_true",
        help="Skip the video remux; emit only final_audio.wav.",
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
        help="What to deliver.",
    )

    # Cache Management
    cache_p = subparsers.add_parser("cache", help="Manage the pipeline cache")
    cache_sub = cache_p.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("info", help="Show cache statistics")
    cache_sub.add_parser("list", help="List all cache entries")
    cache_sub.add_parser("prune", help="Remove stale cache entries")
    cache_sub.add_parser("clear", help="Remove all cache entries")

    # Glossary Management
    gloss_p = subparsers.add_parser("glossary", help="Manage entity glossaries")
    gloss_sub = gloss_p.add_subparsers(dest="glossary_command", required=True)
    gloss_tmpl = gloss_sub.add_parser("template", help="Generate a glossary template")
    gloss_tmpl.add_argument("--output", "-o", default="entity_glossary.json", help="Where to save the template")

    return p


def _resolve_emit(args, src_suffix: str) -> str:
    """Map CLI flags to a concrete emit plan.

    Returns one of: ``audio``, ``video``, ``both``.
    """
    # --audio-only and --emit=audio are equivalent; either wins.
    if getattr(args, "audio_only", False) or getattr(args, "no_video", False) or getattr(args, "emit", "auto") == "audio":
        return "audio"
    if getattr(args, "emit", "auto") == "video":
        return "video"
    # auto / both -> default behaviour: produce video (and audio)
    return "video"


def handle_cache(args):
    cm = CacheManager()
    if args.cache_command == "info":
        stats = cm.get_stats()
        print("Cache Information:")
        print(f"  Root         : {stats['root']}")
        print(f"  Entries      : {stats['count']}")
        print(f"  Total Size   : {stats['total_size_bytes'] / (1024*1024):.2f} MB")
        if stats['newest']:
            import datetime
            newest = datetime.datetime.fromtimestamp(stats['newest']).strftime('%Y-%m-%d %H:%M:%S')
            oldest = datetime.datetime.fromtimestamp(stats['oldest']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Newest Entry : {newest}")
            print(f"  Oldest Entry : {oldest}")
    
    elif args.cache_command == "list":
        entries = cm.list_entries()
        if not entries:
            print("No cache entries found.")
            return
        
        import datetime
        print(f"{'Key':<12} {'Created':<20} {'Size':<10} {'Source'}")
        print("-" * 80)
        for e in entries:
            dt = datetime.datetime.fromtimestamp(e['created_at']).strftime('%Y-%m-%d %H:%M:%S')
            size = f"{e['size_bytes'] / (1024*1024):.1f} MB"
            print(f"{e['cache_key'][:10]:<12} {dt:<20} {size:<10} {Path(e['media_path']).name}")

    elif args.cache_command == "prune":
        count = cm.prune()
        print(f"Pruned {count} stale cache entries.")

    elif args.cache_command == "clear":
        count = cm.clear()
        print(f"Cleared all {count} cache entries.")


def handle_glossary(args):
    if args.glossary_command == "template":
        template = {
            "Ei Nerd": {"action": "preserve"},
            "Spider-Man": {"action": "preserve"},
            "Hulk": {"action": "preserve"},
            "Marvel": {"action": "preserve"},
            "Peter Parker": {"action": "preserve"},
            "Tokeniza": {"action": "preserve"},
            "Mister Negative": {"action": "preserve"},
            "Kurzgesagt": {"action": "preserve"},
            "YouTube": {"action": "preserve"},
            "Gemini": {"action": "preserve"}
        }
        out_path = Path(args.output)
        out_path.write_text(json.dumps(template, indent=2, ensure_ascii=False))
        print(f"Generated glossary template: {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    # Pre-process argv to inject 'run' if no subcommand is present
    if argv is None:
        argv = sys.argv[1:]
    
    if argv and argv[0] not in ("run", "cache", "glossary", "-h", "--help"):
        argv = ["run"] + argv
    elif not argv:
        argv = ["--help"]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "cache":
        handle_cache(args)
        return 0

    if args.command == "glossary":
        handle_glossary(args)
        return 0

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
        workdir=Path(args.workdir).resolve() if args.workdir else None,
        output_path=Path(args.output_dir).resolve(),
        whisper_model=args.whisper_model,
        hf_token=args.hf_token,
        target_lufs=args.target_lufs,
        skip_video=skip_video,
        no_cache=args.no_cache,
        from_stage=args.from_stage,
        read_only_cache=args.read_only_cache,
        adapt_model=args.adapt_model,
        adapt_mode=args.adapt_mode,
        glossary_path=Path(args.glossary).resolve() if args.glossary else None,
    )
    try:
        pipeline.run(start_from=args.start_from, only=args.only)
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    out_dir = Path(args.output_dir).resolve()
    final_audio = out_dir / "final_audio.wav"
    final_video = out_dir / "final_video.mp4"
    # When invoked from dub.sh the success banner is produced by the
    # wrapper; suppress it here to avoid double-printing.
    if not os.environ.get("DUB_SHOW_BANNER"):
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
