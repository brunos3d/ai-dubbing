#!/usr/bin/env python
"""CLI entry point for the pipeline self-optimization system.

Runs an autonomous, resumable optimization loop that repeatedly processes the
same video through the dubbing pipeline (OmniVoice backend), measuring how
closely the generated output matches the source, and searching for the
parameter configuration that maximises that measured quality.

The intended benchmark is **same-language** dubbing (e.g. ``--language en`` on
an English clip), where source and target are identical so timing, speaker
identity, prosody and reconstruction can all be compared directly.

Examples
--------
    # 30 real iterations on the Elon/Colbert clip, English→English
    python scripts/optimize_pipeline.py \
        --media input/elon-musk-might-be-a-super-villain.mp4 \
        --language en --iterations 30

    # Resume the same run later (continues from persisted best/state)
    python scripts/optimize_pipeline.py \
        --media input/elon-musk-might-be-a-super-villain.mp4 --language en

    # Inspect the current best configuration / progression without running
    python scripts/optimize_pipeline.py --media <...> --language en --report

    # Dry run: validate the loop end-to-end with a synthetic scorer (no GPU)
    python scripts/optimize_pipeline.py --media <...> --language en --mock --iterations 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow ``python scripts/optimize_pipeline.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.evaluator import PipelineEvaluator  # noqa: E402
from src.optimization.optimizer import Optimizer, OptimizerConfig  # noqa: E402
from src.optimization.parameter_space import DEFAULT_SPACE  # noqa: E402


def _load_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN"):
                _, _, val = line.partition("=")
                return val.strip().strip('"').strip("'") or None
    return None


def _mock_evaluator(cfg: OptimizerConfig) -> PipelineEvaluator:
    """A GPU-free evaluator with a deterministic synthetic objective.

    The score rewards parameter values near a hidden optimum, so the search
    loop's convergence, persistence and resume can be validated end-to-end
    without running the real pipeline.
    """
    from src.optimization.evaluator import EvaluationResult

    space = DEFAULT_SPACE
    optimum = {
        "samples.target_seconds": 10.0,
        "samples.max_seconds": 15.0,
        "generate.duration_tolerance": 0.10,
        "generate.max_speed": 1.35,
        "generate.max_fit_iters": 2,
        "generate.use_clone_prompt": True,
        "align.tolerance": 0.125,
        "align.min_abs_correction_s": 0.15,
    }

    ev = PipelineEvaluator(
        input_path=cfg.input_path,
        source_language=cfg.language,
        target_language=cfg.language,
        baseline_config=space.coerce_config(space.default_config()),
        to_stage=cfg.to_stage,
        runner=lambda config, from_stage, to_stage: Path(cfg.resolved_run_dir()),
        workspace_root=cfg.resolved_run_dir(),
    )

    def _synthetic_eval(config):  # noqa: ANN001
        dist = 0.0
        for k, v in config.items():
            p = space.get(k)
            if p.kind == "bool":
                dist += 0.0 if bool(v) == bool(optimum[k]) else 1.0
            else:
                span = max(1e-9, p.high - p.low)
                dist += abs(float(v) - float(optimum[k])) / span
        score = max(0.0, 1.0 - dist / max(1, len(config)))
        return EvaluationResult(
            status="ok", score=round(score, 5),
            metrics={"synthetic": round(score, 5)}, from_stage="generate",
        )

    ev.evaluate = _synthetic_eval  # type: ignore[assignment]
    return ev


def _status_dict(cfg: OptimizerConfig) -> dict:
    """A rich, machine-readable status snapshot for the bash wrapper.

    Combines the history summary with the persisted workspace root and the most
    recent iteration, plus the current reconstructed-output file (path / size /
    mtime). Safe to call against a run that is still in progress.
    """
    from src.optimization.history import HistoryStore

    store = HistoryStore(cfg.resolved_run_dir())
    summary = store.summary()
    state = store.load_state()
    records = store.records()
    last = records[-1] if records else None

    ws = state.get("workspace_root") or ""
    out_info = None
    if ws:
        out = Path(ws) / "output" / "reconstructed_speech.wav"
        if out.exists():
            st = out.stat()
            out_info = {
                "path": str(out),
                "size_bytes": st.st_size,
                "mtime": __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            }

    summary.update({
        "run_dir": str(cfg.resolved_run_dir()),
        "workspace_root": ws,
        "language": state.get("language") or cfg.language,
        "stale_steps": state.get("stale_steps"),
        "updated_at": state.get("updated_at"),
        "last_iteration": (last.to_dict() if last else None),
        "current_output": out_info,
    })
    return summary


def _cmd_promote(best_path: Path) -> int:
    """Promote a best.json configuration to the pipeline registry."""
    from src.utils.config import load_pipeline_defaults, save_pipeline_defaults, pipeline_defaults_path
    import shutil

    if not best_path.exists():
        print(f"Error: {best_path} not found.", file=sys.stderr)
        return 1

    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: Failed to load {best_path}: {exc}", file=sys.stderr)
        return 1

    params = best.get("parameters")
    if not params:
        print(f"Error: No parameters found in {best_path}.", file=sys.stderr)
        return 1

    old_defaults = load_pipeline_defaults()
    
    # Backup
    defaults_path = pipeline_defaults_path()
    if defaults_path.exists():
        backup_path = defaults_path.with_suffix(".json.bak")
        shutil.copy2(defaults_path, backup_path)
        print(f"Backup created: {backup_path}")

    save_pipeline_defaults(params)
    print(f"Updated defaults in {defaults_path}")
    print("\nChanges:")
    
    # Diff
    all_keys = sorted(set(old_defaults.keys()) | set(params.keys()))
    for k in all_keys:
        old_v = old_defaults.get(k)
        new_v = params.get(k)
        if old_v != new_v:
            print(f"  {k}:")
            print(f"    {old_v} -> {new_v}")
            
    return 0


def _cmd_show_best(run_dir: Path) -> int:
    """Print the best result from an optimization run."""
    best_path = run_dir / "best.json"
    if not best_path.exists():
        print(f"No best.json found in {run_dir}. Run some iterations first.")
        return 1

    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: Failed to load {best_path}: {exc}")
        return 1

    print(f"Best score: {best.get('score'):.5f}")
    print(f"Iteration : {best.get('iteration')}")
    print(f"Timestamp : {best.get('timestamp')}")
    print("Parameters:")
    params = best.get("parameters", {})
    for k, v in sorted(params.items()):
        print(f"  {k:<30}: {v}")

    # Discover artifacts
    state_path = run_dir / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            ws_root = state.get("workspace_root")
            if ws_root:
                print(f"Workspace : {ws_root}")
                out_audio = Path(ws_root) / "output" / "reconstructed_speech.wav"
                if out_audio.exists():
                    print(f"Output Audio: {out_audio}")
                out_video = Path(ws_root) / "output" / "final_video.mp4"
                if out_video.exists():
                    print(f"Output Video: {out_video}")
        except Exception:
            pass

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media", help="path to the source video/audio")
    ap.add_argument("--language", default="en", help="same-language benchmark code (source==target)")
    ap.add_argument("--iterations", type=int, default=0,
                    help="number of iterations; 0 or negative = infinite (until interrupted)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--run-name", default=None, help="optimization run directory name")
    ap.add_argument("--to-stage", default="reconstruct", help="last stage to run per iteration")
    ap.add_argument("--epsilon", type=float, default=0.2, help="random-exploration probability")
    ap.add_argument("--restart-patience", type=int, default=8)
    ap.add_argument("--early-stop-patience", type=int, default=None)
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--no-pyannote", action="store_true")
    ap.add_argument("--mock", action="store_true", help="synthetic scorer; no GPU/pipeline")
    ap.add_argument("--report", action="store_true", help="print current run summary and exit")
    ap.add_argument("--promote", help="path to best.json to promote to defaults")
    ap.add_argument("--show-best", help="path to optimization run directory")
    args = ap.parse_args()

    if args.promote:
        return _cmd_promote(Path(args.promote))

    if args.show_best:
        return _cmd_show_best(Path(args.show_best))

    if not args.media:
        ap.error("--media is required unless using --promote or --show-best")

    cfg = OptimizerConfig(
        input_path=args.media,
        language=args.language,
        iterations=args.iterations,
        run_name=args.run_name,
        seed=args.seed,
        to_stage=args.to_stage,
        hf_token=_load_hf_token(),
        whisper_model=args.whisper_model,
        no_pyannote=args.no_pyannote,
        epsilon=args.epsilon,
        restart_patience=args.restart_patience,
        early_stop_patience=args.early_stop_patience,
    )

    if args.report:
        print(json.dumps(_status_dict(cfg), indent=2))
        return 0

    optimizer = Optimizer(cfg, evaluator=_mock_evaluator(cfg) if args.mock else None)
    summary = optimizer.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
