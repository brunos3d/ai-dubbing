"""Standalone Pyannote authentication / access verifier.

Isolates Hugging Face authorisation issues from the rest of the dubbing
pipeline.  Run this directly to confirm whether the HF_TOKEN present in
``.env`` grants access to the gated pyannote models used by the pipeline.

Usage::

    .venv/bin/python scripts/test_pyannote_auth.py

Exit code:

* 0 - all required models are reachable
* 1 - one or more models failed (details printed)
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception as exc:  # pragma: no cover - python-dotenv is mandatory
    print(f"ERROR: could not load python-dotenv: {exc}")
    sys.exit(1)


PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"


def _mask(token: str | None) -> str:
    if not token:
        return "<missing>"
    if len(token) < 8:
        return "***"
    return f"{token[:3]}...{token[-3:]} (len={len(token)})"


def _resolve_token() -> Tuple[str | None, str]:
    """Return (token, source) - source is one of 'cli-arg', '.env', 'env', 'none'."""
    if len(sys.argv) > 1 and sys.argv[1]:
        return sys.argv[1], "cli-arg"
    token = os.environ.get("HF_TOKEN")
    if token:
        return token, "env"
    return None, "none"


def _check_whoami(api) -> Tuple[bool, str]:
    try:
        me = api.whoami()
        return True, f"whoami ok: {me.get('name', '<unknown>')} ({me.get('fullname', '')})"
    except Exception as exc:
        return False, f"whoami failed: {exc}"


def _check_model_info(api, repo_id: str) -> Tuple[bool, str]:
    """Lightweight probe: returns whether model metadata is reachable.

    Note: ``model_info`` does NOT require the user to have accepted
    gating.  A gated repo will still report ``gated='auto'`` from this
    call.  The decisive test is whether ``hf_hub_download`` succeeds.
    """
    try:
        info = api.model_info(repo_id)
        return True, f"model_info ok (gated={info.gated}, sha={info.sha[:8] if info.sha else '?'})"
    except Exception as exc:
        return False, f"model_info failed: {exc}"


def _check_model_download(api, repo_id: str, filename: str) -> Tuple[bool, str]:
    """Decisive probe: actually fetch a file from the repo."""
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=filename)
        return True, f"download ok: {filename} -> {path}"
    except Exception as exc:
        msg = str(exc)
        first_line = msg.split("\n", 1)[0][:240]
        return False, f"download failed ({filename}): {first_line}"


def _check_pyannote_pipeline(token: str | None) -> Tuple[bool, str]:
    """Try the actual pyannote path.  This is what the pipeline does."""
    try:
        pyannote_audio = importlib.import_module("pyannote.audio")
    except Exception as exc:
        return False, f"pyannote.audio not importable: {exc}"
    try:
        Pipeline = pyannote_audio.Pipeline
    except AttributeError as exc:
        return False, f"pyannote.audio.Pipeline missing: {exc}"

    # Try the modern API first (4.x) and fall back to the legacy name (3.x).
    last_exc: Exception | None = None
    try:
        try:
            pipe = Pipeline.from_pretrained(PIPELINE_MODEL, token=token)
        except TypeError:
            pipe = Pipeline.from_pretrained(PIPELINE_MODEL, use_auth_token=token)
        if pipe is None:
            return False, "Pipeline.from_pretrained returned None (auth likely failed silently)"
        return True, f"Pipeline loaded: {type(pipe).__name__}"
    except Exception as exc:
        last_exc = exc
        return False, f"Pipeline.from_pretrained raised: {type(exc).__name__}: {exc}"


def main() -> int:
    print("=" * 64)
    print("Pyannote / Hugging Face access verification")
    print("=" * 64)

    token, source = _resolve_token()
    print(f"HF_TOKEN source : {source}")
    print(f"HF_TOKEN value  : {_mask(token)}")
    print(f"Python          : {sys.version.split()[0]}")
    print()

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    except Exception as exc:
        print(f"FAIL: huggingface_hub import failed: {exc}")
        return 1

    print("--- Token reachability ---")
    ok, msg = _check_whoami(api)
    print(f"{'PASS' if ok else 'FAIL'}: {msg}")
    print()

    if not ok:
        print("Token validation failed; downstream checks skipped.")
        return 1

    print("--- Model metadata ---")
    for repo in (PIPELINE_MODEL, SEGMENTATION_MODEL):
        ok, msg = _check_model_info(api, repo)
        print(f"{'PASS' if ok else 'FAIL'}: {repo} -> {msg}")
    print()

    print("--- Decisive download probe ---")
    # We only need to confirm a file we can access for the *pipeline* repo,
    # and the model weights for the *segmentation* repo, which is the one
    # that is gated.
    decisive_checks = [
        (PIPELINE_MODEL, "config.yaml"),
        (SEGMENTATION_MODEL, "config.yaml"),
        (SEGMENTATION_MODEL, "pytorch_model.bin"),
    ]
    decisive_results: list[bool] = []
    for repo, fname in decisive_checks:
        ok, msg = _check_model_download(api, repo, fname)
        decisive_results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}: {repo} :: {msg}")
    print()

    print("--- Pyannote pipeline load ---")
    ok, msg = _check_pyannote_pipeline(token)
    print(f"{'PASS' if ok else 'FAIL'}: {msg}")
    print()

    print("=" * 64)
    if all(decisive_results) and ok:
        print("RESULT: PASS - all required models reachable.")
        return 0
    print("RESULT: FAIL - one or more models are not accessible.")
    print(
        "If you see a 403/GatedRepoError, the HF account behind the token\n"
        "has not been granted access to the gated pyannote models.  Visit\n"
        "  https://huggingface.co/pyannote/segmentation-3.0\n"
        "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "and accept the user conditions, then re-run this script."
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
