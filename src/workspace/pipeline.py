"""Workspace-aware pipeline orchestrator (spec §5, §6, §9).

This module is the durable home of the workspace architecture: a
:class:`WorkspacePipeline` wraps the existing legacy :class:`Pipeline` with
two new high-level operations:

* :meth:`WorkspacePipeline.prepare` — runs ``extract .. translate`` and stops.
* :meth:`WorkspacePipeline.generate` — runs the dirty subset of
  ``generate .. video`` based on the invalidation DAG in
  :mod:`src.workspace.dag`.

The rest of the file is the bookkeeping that ties the two halves together:

* Module-level stage registry and the stage-to-subdir map (spec §5).
* :class:`WorkspaceContext` — a dataclass that's *also* a dict, so the existing
  ``context["audio_path"]`` lookups in the legacy stages keep working.
* :meth:`WorkspacePipeline._run_single_stage` — the core "stage against a
  staging dir, promote on success, record artefacts on success, mark failed on
  exception" loop (spec §11).

Tests live in ``tests/test_workspace_pipeline_helpers.py`` (helper tests),
``tests/test_workspace_e2e.py`` (full prepare+generate), and
``tests/test_workspace_edit_scenarios.py`` (the eight edit scenarios from
spec §10).
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..stages import (
    AlignStage,
    DiarizeStage,
    ExtractStage,
    GenerateStage,
    MixStage,
    ReconstructStage,
    SampleStage,
    SeparateStage,
    TranslateStage,
    TranscribeStage,
    VideoStage,
)
from ..utils.cache import media_hash as _media_hash
from ..utils.logging import get_logger
from .atomic import promote, sha256_file, stage_staging_dir
from .content_hash import content_hash as _content_hash, pipeline_config_hash
from .dag import CliOverrides, STAGE_ORDER, compute_stale_set
from .manifest import ArtifactRef, Manifest, StageRecord
from .paths import slugify, workspace_id, workspaces_root

LOG = get_logger("ai-dubbing.workspace")

# ---------------------------------------------------------------------------
# Module-level stage registry and helpers
# ---------------------------------------------------------------------------

STAGE_CLASSES: Dict[str, Any] = {
    "extract": ExtractStage,
    "separate": SeparateStage,
    "diarize": DiarizeStage,
    "samples": SampleStage,
    "transcribe": TranscribeStage,
    "translate": TranslateStage,
    "generate": GenerateStage,
    "align": AlignStage,
    "reconstruct": ReconstructStage,
    "mix": MixStage,
    "video": VideoStage,
}

# Per-stage workspace-relative subdir. ``None`` means the stage writes
# directly into the workspace root. Names are spec §5 — the canonical layout
# in :func:`workspace_layout` mirrors them.
STAGE_SUBDIR: Dict[str, Optional[str]] = {
    "extract": "media",
    "separate": "media",
    "diarize": "diarization",
    "samples": None,
    "transcribe": "transcription",
    "translate": "translation",
    "generate": None,
    "align": None,
    "reconstruct": None,
    "mix": None,
    "video": None,
}

# Editable artefacts (spec §9). These are the paths a user can hand-edit and
# still expect the next :meth:`WorkspacePipeline.generate` to pick up the
# change.
_EDITABLE_PATHS: List[str] = [
    "transcription/transcript.json",
    "translation/translated_transcript.json",
    "translation/glossary.json",
    "speakers/*/primary.wav",
    "speakers/*/primary.txt",
]

# Derived artefacts (spec §9). These are intentionally excluded from
# invalidation: they are regenerated as a side-effect of the stage that owns
# them.
_DERIVED_PATHS: List[str] = [
    "diarization/embeddings.npz",
    "logs/pipeline.log",
]

# Well-known context keys (spec §6) that all downstream stages read. The
# values are pre-populated before the first stage runs and refreshed in
# :meth:`WorkspacePipeline._run_stages` between stages.
_CONTEXT_KEYS: Tuple[str, ...] = (
    "audio_path",
    "speech_path",
    "background_path",
    "segments_path",
    "transcript_path",
    "translated_path",
    "generated_dir",
    "manifest_path",
    "aligned_dir",
    "aligned_manifest",
    "reconstructed_path",
    "final_path",
    "final_video",
)


def stage_subdir(name: str) -> Optional[str]:
    """Return the workspace-relative subdir for stage ``name``, or ``None``."""
    return STAGE_SUBDIR.get(name)


def prepare_stages() -> List[str]:
    """Return the canonical ``extract .. translate`` list."""
    return list(STAGE_ORDER[: STAGE_ORDER.index("translate") + 1])


def generate_stages() -> List[str]:
    """Return the canonical ``generate .. video`` list."""
    return list(STAGE_ORDER[STAGE_ORDER.index("generate") :])


def workspace_layout() -> List[str]:
    """Return the canonical list of top-level subdirs a workspace owns.

    Order is roughly temporal: extracted sources first, derived artefacts in
    the middle, user-visible outputs at the end.
    """
    return [
        "media",
        "diarization",
        "transcription",
        "translation",
        "speakers",
        "output",
        "source",
        "logs",
    ]


# ---------------------------------------------------------------------------
# Stage factory
# ---------------------------------------------------------------------------


def _build_stage(
    name: str,
    workspace_root: Path,
    *,
    no_pyannote: bool,
    hf_token: Optional[str],
    whisper_model: str,
    source_language: str,
    target_language: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    glossary_path: Optional[Path],
    target_lufs: float,
):
    """Instantiate the right stage class with the right constructor kwargs.

    Passes ``subdir=STAGE_SUBDIR[name]`` so each stage's ``self.workdir`` is
    the nested ``<workspace_root>/<subdir>`` (or just ``<workspace_root>``
    when ``subdir`` is ``None``). The ``output_dir`` for the synthesis stages
    is the workspace's ``output/`` subdir, and the ``input_path`` for the
    video stage is the original media (re-symlinked under ``source/`` by
    :meth:`WorkspacePipeline.prepare`).
    """
    subdir = stage_subdir(name)
    output_dir = workspace_root / "output"

    if name == "extract":
        return ExtractStage(workspace_root, subdir=subdir)
    if name == "separate":
        return SeparateStage(workspace_root, subdir=subdir)
    if name == "diarize":
        return DiarizeStage(
            workspace_root,
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            no_pyannote=no_pyannote,
            subdir=subdir,
        )
    if name == "samples":
        return SampleStage(workspace_root, subdir=subdir)
    if name == "transcribe":
        return TranscribeStage(
            workspace_root,
            model_size=whisper_model,
            source_language=source_language,
            subdir=subdir,
        )
    if name == "translate":
        return TranslateStage(
            workspace_root,
            source_language,
            target_language,
            glossary_path=glossary_path,
            subdir=subdir,
        )
    if name == "generate":
        return GenerateStage(workspace_root, target_language=target_language, subdir=subdir)
    if name == "align":
        return AlignStage(workspace_root, target_language=target_language, subdir=subdir)
    if name == "reconstruct":
        return ReconstructStage(workspace_root, output_dir, subdir=subdir)
    if name == "mix":
        return MixStage(workspace_root, output_dir, target_lufs=target_lufs, subdir=subdir)
    if name == "video":
        return VideoStage(
            workspace_root,
            output_dir,
            str(workspace_root / "source"),
            subdir=subdir,
        )
    raise KeyError(f"unknown stage: {name!r}")


# ---------------------------------------------------------------------------
# WorkspaceContext
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceContext:
    """A dict-like bag of paths and language codes that the stages read.

    Existing stages look up ``context["audio_path"]``, ``context["workdir"]``,
    etc. — we keep that contract by delegating the dict operations to
    ``self._data``. We also expose the well-known attributes as proper fields
    so type-checkers and the rest of the orchestrator can refer to them
    without spelling the keys.
    """

    workspace_root: Path
    input_path: str
    source_language: str
    target_language: str
    workdir: str = ""
    output_dir: str = ""
    _data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workdir:
            self.workdir = str(self.workspace_root)
        if not self.output_dir:
            self.output_dir = str(self.workspace_root / "output")
        if self._data is None:
            self._data = {}

    # Dict delegation -------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


# ---------------------------------------------------------------------------
# WorkspacePipeline
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (with ``Z``)."""
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    """Return the short git commit of the repo, or ``"unknown"`` if unavailable."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        )
        return out.strip() or "unknown"
    except Exception:  # noqa: BLE001 - any failure is "unknown"
        return "unknown"


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class WorkspacePipeline:
    """The two-phase workspace orchestrator (spec §5)."""

    def __init__(
        self,
        input_path: str,
        source_language: str,
        target_language: str,
        *,
        whisper_model: str = "large-v3",
        hf_token: Optional[str] = None,
        target_lufs: float = -16.0,
        glossary_path: Optional[Path] = None,
        no_pyannote: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        workspace_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.input_path = input_path
        self.source_language = source_language
        self.target_language = target_language
        self.whisper_model = whisper_model
        self.hf_token = hf_token
        self.target_lufs = target_lufs
        self.glossary_path = glossary_path
        self.no_pyannote = no_pyannote
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._workspace_root_override = workspace_root
        self._output_dir_override = output_dir

    # -- configuration / identity helpers --------------------------------

    def _config(self) -> Dict[str, Any]:
        """Return the pipeline config dict (mirrors ``Pipeline.pipeline_config_dict``)."""
        return {
            "whisper_model": self.whisper_model,
            "target_lufs": self.target_lufs,
            "no_pyannote": self.no_pyannote,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "skip_video": False,
            "glossary_path": str(self.glossary_path) if self.glossary_path else None,
            "hf_token_available": bool(self.hf_token),
        }

    def _media_hash(self) -> str:
        """Return the SHA-256 of the input media file."""
        return _media_hash(self.input_path)

    def _id(self) -> Tuple[str, str, str]:
        """Compute ``(workspace_id, content_hash, pipeline_config_hash)``."""
        cfg_hash = pipeline_config_hash(self._config())
        media_h = self._media_hash()
        ch = _content_hash(
            media_sha256=media_h,
            src_lang=self.source_language,
            tgt_lang=self.target_language,
            pipeline_config_hash=cfg_hash,
        )
        slug = slugify(Path(self.input_path).stem) or "workspace"
        wid = workspace_id(
            media_sha256=media_h,
            src_lang=self.source_language,
            tgt_lang=self.target_language,
            pipeline_config_hash=cfg_hash,
            source_slug=slug,
        )
        return wid, ch, cfg_hash

    # -- layout -----------------------------------------------------------

    def _ensure_layout(self, root: Path) -> None:
        """Create every canonical subdir under ``root`` (idempotent)."""
        root = Path(root)
        for d in workspace_layout():
            (root / d).mkdir(parents=True, exist_ok=True)

    def _resolve_workspace_root(self, wid: str) -> Path:
        if self._workspace_root_override is not None:
            return Path(self._workspace_root_override)
        return workspaces_root() / wid

    # -- manifest loading ------------------------------------------------

    def _load_or_init_manifest(
        self, workspace_id_str: str, root: Optional[Path] = None
    ) -> Manifest:
        """Load ``<root>/manifest.json`` if it exists; otherwise create a fresh one."""
        if root is None:
            root = self._resolve_workspace_root(workspace_id_str)
        manifest_path = root / "manifest.json"
        existing = Manifest.load(manifest_path)
        if existing is not None:
            return existing
        m = Manifest.create(
            workspace_id=workspace_id_str,
            pipeline_version="0.1.0",
            git_commit=_git_commit(),
        )
        for p in _EDITABLE_PATHS:
            m.add_editable_path(p)
        for p in _DERIVED_PATHS:
            m.add_derived_path(p)
        # Seed an empty StageRecord for every stage so downstream code can
        # always look up ``manifest.stages[name]``.
        for name in STAGE_ORDER:
            m.add_stage(name, StageRecord(name=name, status="pending"))
        return m

    # -- prepare / generate ----------------------------------------------

    def prepare(self) -> Tuple[str, Path]:
        """Run the ``extract .. translate`` stages; return ``(workspace_id, root)``."""
        wid, content_h, cfg_hash = self._id()
        root = self._resolve_workspace_root(wid)
        root.mkdir(parents=True, exist_ok=True)
        self._ensure_layout(root)

        self._link_source(root)
        self._write_metadata(root, content_h, cfg_hash)

        manifest = self._load_or_init_manifest(wid, root=root)
        self._run_stages(manifest, prepare_stages(), start_at="extract", workspace_root=root)
        return wid, root

    def generate(
        self,
        workspace_id_str: Optional[str] = None,
        *,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[str, Path]:
        """Run the dirty subset of ``generate .. video``; return ``(workspace_id, root)``."""
        if workspace_id_str is None:
            workspace_id_str, _, _ = self._id()
        root = self._resolve_workspace_root(workspace_id_str)
        manifest_path = root / "manifest.json"
        manifest = Manifest.load(manifest_path)
        if manifest is None:
            raise FileNotFoundError(
                f"No manifest.json at {manifest_path}; run prepare() first."
            )

        overrides = CliOverrides(force=force, from_stage=from_stage, to_stage=to_stage)
        stale = compute_stale_set(manifest, root, overrides)
        if not stale:
            LOG.info("Nothing to do: no stale stages for %s", workspace_id_str)
            return workspace_id_str, root
        self._run_stages(manifest, stale, start_at=stale[0], workspace_root=root)
        return workspace_id_str, root

    # -- metadata.json ----------------------------------------------------

    def _link_source(self, root: Path) -> None:
        """Symlink the original input under ``<root>/source/<basename>``.

        Best-effort: if symlink creation fails (e.g. unsupported filesystem or
        a pre-existing symlink with the wrong target), we just log and move
        on; the manifest will still record the right ``media_path``.
        """
        src = Path(self.input_path).resolve()
        if not src.exists():
            return
        dest = root / "source" / src.name
        if dest.exists() or dest.is_symlink():
            try:
                dest.unlink()
            except OSError:
                pass
        try:
            dest.symlink_to(src)
        except OSError as exc:
            LOG.warning("Could not symlink source %s -> %s: %s", src, dest, exc)

    def _write_metadata(self, root: Path, content_h: str, cfg_hash: str) -> None:
        """Write ``<root>/metadata.json`` per spec §8 (idempotent)."""
        meta_path = root / "metadata.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = None
            if existing and existing.get("workspace_id") == _read_workspace_id(root):
                return

        wid = _read_workspace_id(root) or root.name
        now = _now_iso()
        payload: Dict[str, Any] = {
            "workspace_schema_version": 1,
            "workspace_id": wid,
            "content_hash": content_h,
            "created_at": now,
            "updated_at": now,
            "workspace_created_with": {
                "pipeline_version": "0.1.0",
                "git_commit": _git_commit(),
                "python_version": _python_version(),
            },
            "source": {
                "media_path": str(Path(self.input_path).resolve()),
                "media_sha256": self._media_hash(),
                "source_language": self.source_language,
                "target_language": self.target_language,
            },
            "config": self._config(),
            "pipeline_config_hash": cfg_hash,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # -- run-stages loop -------------------------------------------------

    def _run_stages(
        self,
        manifest: Manifest,
        stages: List[str],
        start_at: str,
        *,
        workspace_root: Path,
    ) -> None:
        """Run ``stages`` in order; pre-populate the context from the spec §6 layout."""
        context = WorkspaceContext(
            workspace_root=workspace_root,
            input_path=self.input_path,
            source_language=self.source_language,
            target_language=self.target_language,
            output_dir=str(workspace_root / "output"),
        )
        # Pre-populate the well-known context keys so stages that read them
        # before :meth:`_run_single_stage` can hydrate them have something
        # sensible to read. We use the workspace-relative paths the spec
        # defines, joined with ``workspace_root`` (and the per-stage subdir
        # where applicable).
        context["input_path"] = str(workspace_root / "source" / Path(self.input_path).name)
        context["source_language"] = self.source_language
        context["target_language"] = self.target_language
        context["workdir"] = str(workspace_root)
        context["output_dir"] = str(workspace_root / "output")
        context["audio_path"] = str(workspace_root / "media" / "original_audio.wav")
        context["speech_path"] = str(workspace_root / "media" / "speech.wav")
        context["background_path"] = str(workspace_root / "media" / "background.wav")
        context["segments_path"] = str(workspace_root / "diarization" / "segments.json")
        context["transcript_path"] = str(workspace_root / "transcription" / "transcript.json")
        context["translated_path"] = str(
            workspace_root / "translation" / "translated_transcript.json"
        )
        context["generated_dir"] = str(workspace_root / "generated_segments")
        context["manifest_path"] = str(workspace_root / "generated_segments" / "manifest.json")
        context["aligned_dir"] = str(workspace_root / "aligned_segments")
        context["aligned_manifest"] = str(workspace_root / "aligned_manifest.json")
        context["reconstructed_path"] = str(workspace_root / "output" / "reconstructed_speech.wav")
        context["final_path"] = str(workspace_root / "output" / "final_audio.wav")
        context["final_video"] = str(workspace_root / "output" / "final_video.mp4")

        # ``stages[0]`` may not equal ``start_at`` when ``--to-stage`` trims
        # the bottom of the range. Skip everything before ``start_at``.
        try:
            start_idx = stages.index(start_at)
        except ValueError:
            start_idx = 0
        for name in stages[start_idx:]:
            self._run_single_stage(manifest, name, context, workspace_root=workspace_root)

    def _run_single_stage(
        self,
        manifest: Manifest,
        name: str,
        context: WorkspaceContext,
        *,
        workspace_root: Path,
    ) -> None:
        """Run a single stage against a staging dir; promote on success.

        Records a ``StageRecord`` with all declared inputs/outputs as
        ``ArtifactRef`` (skipped for files that don't exist on disk after the
        promote). On exception, the staging dir is removed, the stage is
        marked ``failed``, and the exception is re-raised.
        """
        stage_cls = STAGE_CLASSES[name]
        staging = stage_staging_dir(workspace_root, name)
        staging.mkdir(parents=True, exist_ok=True)
        started = time.time()
        started_at = _now_iso()

        stage = _build_stage(
            name,
            workspace_root,
            no_pyannote=self.no_pyannote,
            hf_token=self.hf_token,
            whisper_model=self.whisper_model,
            source_language=self.source_language,
            target_language=self.target_language,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            glossary_path=self.glossary_path,
            target_lufs=self.target_lufs,
        )
        # Redirect the stage's writes into the staging dir.
        subdir = stage_subdir(name)
        stage.workdir = staging / subdir if subdir else staging
        if subdir:
            stage.workdir.mkdir(parents=True, exist_ok=True)
        # For stages that need an output_dir, also point it into staging.
        if name in {"reconstruct", "mix", "video"}:
            stage.output_dir = staging / "output"
            stage.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if name == "extract":
                stage.run(self.input_path, context._data)
            else:
                stage.run(context._data)
        except Exception as exc:  # noqa: BLE001 - we record and re-raise
            shutil.rmtree(staging, ignore_errors=True)
            failed = StageRecord(
                name=name,
                status="failed",
                started_at=started_at,
                finished_at=_now_iso(),
                duration_s=time.time() - started,
                config={
                    k: getattr(self, k) for k in getattr(stage_cls, "config_fields", [])
                },
                error=str(exc),
            )
            manifest.add_stage(name, failed)
            manifest.save(workspace_root / "manifest.json")
            LOG.error("Stage %s failed: %s", name, exc)
            raise

        # On success, promote staging into the workspace root.
        promote(staging, workspace_root)

        # Build the per-stage record: config, inputs, outputs (with hashes).
        config = {
            k: getattr(self, k) for k in getattr(stage_cls, "config_fields", [])
        }
        inputs: List[ArtifactRef] = []
        for rel in getattr(stage_cls, "inputs", []):
            ref = self._maybe_artifact(workspace_root, rel)
            if ref is not None:
                inputs.append(ref)
        outputs: List[ArtifactRef] = []
        for rel in getattr(stage_cls, "outputs", []):
            ref = self._maybe_artifact(workspace_root, rel)
            if ref is not None:
                outputs.append(ref)

        manifest.add_stage(
            name,
            StageRecord(
                name=name,
                status="done",
                started_at=started_at,
                finished_at=_now_iso(),
                duration_s=time.time() - started,
                config=config,
                inputs=inputs,
                outputs=outputs,
            ),
        )
        manifest.save(workspace_root / "manifest.json")
        LOG.info("[ok] %s done in %.2fs", name, time.time() - started)

    @staticmethod
    def _maybe_artifact(workspace_root: Path, rel_path: str) -> Optional[ArtifactRef]:
        """Return an ``ArtifactRef`` for ``rel_path`` if it exists on disk."""
        full = Path(workspace_root) / rel_path
        if not full.exists() or not full.is_file():
            return None
        try:
            size = full.stat().st_size
            digest = sha256_file(full)
        except OSError:
            return None
        return ArtifactRef(path=rel_path, sha256=digest, size_bytes=size)

    def _build_stage_for_staging(
        self, name: str, staging: Path, context: WorkspaceContext
    ):
        """Factory used by tests and callers that want to build a stage
        already pointed at a custom ``workdir``.

        Currently unused by the production run loop (which calls
        :func:`_build_stage` directly so it can set ``subdir`` before
        redirecting ``workdir`` to ``staging``), but exposed for symmetry with
        the plan.
        """
        stage = _build_stage(
            name,
            context.workspace_root,
            no_pyannote=self.no_pyannote,
            hf_token=self.hf_token,
            whisper_model=self.whisper_model,
            source_language=self.source_language,
            target_language=self.target_language,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            glossary_path=self.glossary_path,
            target_lufs=self.target_lufs,
        )
        subdir = stage_subdir(name)
        stage.workdir = staging / subdir if subdir else staging
        return stage


# ---------------------------------------------------------------------------
# Tiny helpers used above
# ---------------------------------------------------------------------------


def _read_workspace_id(root: Path) -> Optional[str]:
    """Try to read the ``workspace_id`` field from an existing metadata.json."""
    meta = root / "metadata.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("workspace_id")
    except (json.JSONDecodeError, OSError):
        return None


# Re-exported helpers for tests / callers.
__all__ = [
    "STAGE_CLASSES",
    "STAGE_SUBDIR",
    "STAGE_ORDER",
    "WorkspaceContext",
    "WorkspacePipeline",
    "generate_stages",
    "prepare_stages",
    "stage_subdir",
    "workspace_layout",
]
