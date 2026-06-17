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
from .timeline import refresh_timeline

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
    "translation/timing_report.json",
    "generated_segments/prosody.json",
    "timeline.json",
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


def _find_source_media(root: Path) -> Optional[Path]:
    """Return the path to the single file under ``root/source/``, if it exists."""
    source_dir = root / "source"
    if not source_dir.exists():
        return None
    files = [p for p in source_dir.iterdir() if p.is_file()]
    return files[0] if files else None


def _rewrite_staging_paths(obj, staging_str: str, workspace_str: str):
    """Recursively replace any string in ``obj`` that starts with
    ``staging_str`` so it starts with ``workspace_str`` instead.

    The orchestrator redirects each stage's writes to a staging dir
    under ``<root>/.tmp/<stage>-<uuid8>/`` and only promotes to the
    real workspace root on success.  Stage return values often embed
    absolute paths inside that staging dir (e.g.
    ``speaker_profiles[spk].profiles.primary.reference`` points to
    ``<staging>/speaker_profiles/.../reference.wav``).  After
    :func:`promote` the staging dir is gone, so downstream stages
    would dereference a stale path.  This helper walks the return
    value and rewrites every string that names a file or directory
    inside the staging dir to point at its post-promote location.
    """
    if isinstance(obj, str):
        if obj.startswith(staging_str):
            return workspace_str + obj[len(staging_str):]
        return obj
    if isinstance(obj, dict):
        return {k: _rewrite_staging_paths(v, staging_str, workspace_str) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_rewrite_staging_paths(v, staging_str, workspace_str) for v in obj)
    return obj


def load_speaker_profiles_from_disk(workspace_root: Path) -> Dict[str, Any]:
    """Rebuild the ``speaker_profiles`` context dict from the on-disk
    ``speaker_profiles/`` directory.

    The samples stage writes one sub-directory per speaker under
    ``<workspace>/speaker_profiles/{spk}/{profile}/`` containing
    ``metadata.json``, ``reference.wav`` and ``transcript.txt``.  The
    OmniVoice generate stage consumes this dict from
    ``context["speaker_profiles"]`` to drive voice cloning.  When
    generate is reached on a resume (samples not re-run in the
    current session), the context is empty unless we re-hydrate it
    from the persisted tree.

    ``metadata.json`` may record absolute paths from a previous run
    (e.g. ``<root>/.tmp/samples-<uuid8>/speaker_profiles/...``) that
    no longer exist because the staging dir was cleaned up after
    :func:`promote`.  We therefore prefer the on-disk canonical path
    ``<profiles_dir>/<spk>/<profile>/reference.wav`` whenever that
    file exists, falling back to the metadata's recorded path only
    when the canonical file is missing.

    Mirrors the legacy ``Pipeline._build_context`` behaviour
    (``src/pipeline.py`` lines 314-334) so the workspace pipeline
    behaves identically.
    """
    profiles_dir = Path(workspace_root) / "speaker_profiles"
    out: Dict[str, Any] = {}
    if not profiles_dir.exists():
        return out
    for spk_path in profiles_dir.iterdir():
        if not spk_path.is_dir():
            continue
        spk = spk_path.name
        out[spk] = {"profiles": {}}
        for prof_path in spk_path.iterdir():
            if not prof_path.is_dir():
                continue
            meta_path = prof_path / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            transcript_path = prof_path / "transcript.txt"
            # Prefer the on-disk canonical reference path.  The
            # metadata's recorded path may point at a cleaned-up
            # staging dir from a previous run.
            canonical_reference = prof_path / "reference.wav"
            if canonical_reference.exists():
                reference = str(canonical_reference)
            else:
                reference = meta.get(
                    "reference_path", str(canonical_reference)
                )
            out[spk]["profiles"][prof_path.name] = {
                "reference": reference,
                "transcript_text": (
                    transcript_path.read_text(encoding="utf-8").strip()
                    if transcript_path.exists()
                    else ""
                ),
                "score": meta.get("score", 0.0),
            }
    return out


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
    tts_backend: str = "omnivoice",
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
        return GenerateStage(
            workspace_root,
            target_language=target_language,
            tts_backend=tts_backend,
            subdir=subdir,
        )
    if name == "align":
        return AlignStage(workspace_root, target_language=target_language, subdir=subdir)
    if name == "reconstruct":
        return ReconstructStage(workspace_root, output_dir, subdir=subdir)
    if name == "mix":
        return MixStage(workspace_root, output_dir, target_lufs=target_lufs, subdir=subdir)
    if name == "video":
        source_media = _find_source_media(workspace_root)
        return VideoStage(
            workspace_root,
            output_dir,
            str(source_media) if source_media else str(workspace_root / "source"),
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
        tts_backend: str = "omnivoice",
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
        # Selected TTS backend. Tracked per-stage (generate.config_fields), so
        # it invalidates generate..video on change but does NOT participate in
        # the workspace-identity hash (extract..translate stay valid).
        self.tts_backend = (tts_backend or "omnivoice").strip().lower()
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

    def _current_configs(self) -> Dict[str, Dict[str, Any]]:
        """Build the per-stage ``current_configs`` map for the invalidation DAG.

        For every stage, collect the values of its declared ``config_fields``
        from this pipeline instance (``None`` for fields the pipeline does not
        carry — the DAG treats ``None`` as "not an override", see
        :func:`src.workspace.dag._stage_has_config_change`).
        """
        configs: Dict[str, Dict[str, Any]] = {}
        for name in STAGE_ORDER:
            cls = STAGE_CLASSES.get(name)
            if cls:
                configs[name] = {
                    k: getattr(self, k, None)
                    for k in getattr(cls, "config_fields", [])
                }
        return configs

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
        # always look up ``manifest.stages[name]`` and so the invalidation
        # DAG can see the declared inputs/outputs of stages that have not
        # yet been run (e.g. ``generate..video`` after ``prepare()``).
        for name in STAGE_ORDER:
            cls = STAGE_CLASSES.get(name)
            if cls is None:
                m.add_stage(name, StageRecord(name=name, status="pending"))
                continue
            inputs = []
            for rel in getattr(cls, "inputs", []):
                ref = self._maybe_artifact(root, rel)
                if ref is not None:
                    inputs.append(ref)
            outputs = []
            for rel in getattr(cls, "outputs", []):
                ref = self._maybe_artifact(root, rel)
                if ref is not None:
                    outputs.append(ref)
            m.add_stage(
                name,
                StageRecord(
                    name=name,
                    status="pending",
                    inputs=inputs,
                    outputs=outputs,
                ),
            )
        return m

    # -- prepare / generate ----------------------------------------------

    def prepare(self, *, force: bool = False) -> Tuple[str, Path]:
        """Run the stale subset of ``extract .. translate``; return ``(workspace_id, root)``.

        ``prepare`` is now staleness-aware (it used to re-run every analysis
        stage unconditionally).  On a fresh workspace every prepare stage is
        stale (its outputs do not exist yet) so all of them run; on a repeat
        ``prepare`` of an unchanged input nothing is stale and the cached
        artefacts are reused — this is what restores the resume/cache
        behaviour the one-shot ``run`` path relies on, now that ``run`` is
        routed through the workspace orchestrator.
        """
        wid, content_h, cfg_hash = self._id()
        root = self._resolve_workspace_root(wid)
        root.mkdir(parents=True, exist_ok=True)
        self._ensure_layout(root)

        self._link_source(root)
        self._write_metadata(root, content_h, cfg_hash)

        manifest = self._load_or_init_manifest(wid, root=root)

        overrides = CliOverrides(force=force)
        stale = compute_stale_set(
            manifest, root, overrides, current_configs=self._current_configs()
        )
        prepare_set = [s for s in prepare_stages() if s in stale]
        if prepare_set:
            self._run_stages(
                manifest, prepare_set, start_at=prepare_set[0], workspace_root=root
            )
        else:
            LOG.info("prepare: all analysis stages up to date for %s", wid)
        # Record the *initial* disk hashes for the not-yet-run generate
        # stages.  This must happen *before* the user has a chance to edit
        # any file, so the invalidation DAG in a later ``generate()`` has
        # a real baseline to diff against.
        self._backfill_manifest_from_disk(manifest, root)
        manifest.save(root / "manifest.json")
        self._refresh_timeline(root)
        return wid, root

    def _refresh_timeline(self, root: Path) -> None:
        """Rebuild the derived ``timeline.json`` (best-effort; never fatal)."""
        try:
            tl = refresh_timeline(root)
            LOG.info("timeline.json refreshed: %s", tl.summary())
        except Exception as exc:  # noqa: BLE001 - the Timeline is derived/optional
            LOG.warning("Could not refresh timeline.json: %s", exc)

    def _hydrate_from_metadata(self, root: Path) -> None:
        """Hydrate pipeline attributes from metadata.json."""
        meta_path = root / "metadata.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        source = meta.get("source", {})
        if self.input_path == "/dev/null":
            self.input_path = source.get("media_path", self.input_path)
        if self.source_language == "?":
            self.source_language = source.get("source_language", self.source_language)
        if self.target_language == "?":
            self.target_language = source.get("target_language", self.target_language)

        config = meta.get("config", {})
        # We only hydrate if the current values are defaults.
        # This is a bit heuristic, but it works for 'generate' CLI.
        if self.whisper_model == "large-v3":
            self.whisper_model = config.get("whisper_model", self.whisper_model)
        if self.target_lufs == -16.0:
            self.target_lufs = config.get("target_lufs", self.target_lufs)
        if self.no_pyannote is False:
            self.no_pyannote = config.get("no_pyannote", self.no_pyannote)
        if self.min_speakers is None:
            self.min_speakers = config.get("min_speakers", self.min_speakers)
        if self.max_speakers is None:
            self.max_speakers = config.get("max_speakers", self.max_speakers)
        if self.glossary_path is None and config.get("glossary_path"):
            self.glossary_path = Path(config["glossary_path"])

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
        
        # Hydrate settings from metadata before computing staleness.
        self._hydrate_from_metadata(root)
        # Persist the (possibly newly-selected) TTS backend into metadata.json.
        # _write_metadata only rewrites when the backend actually changed.
        try:
            _, content_h, cfg_hash = self._id()
            self._write_metadata(root, content_h, cfg_hash)
        except Exception as exc:  # noqa: BLE001 - metadata refresh is best-effort
            LOG.debug("metadata backend refresh skipped: %s", exc)

        manifest_path = root / "manifest.json"
        manifest = Manifest.load(manifest_path)
        if manifest is None:
            raise FileNotFoundError(
                f"No manifest.json at {manifest_path}; run prepare() first."
            )

        # Build current_configs for invalidation DAG.
        current_configs = self._current_configs()

        overrides = CliOverrides(force=force, from_stage=from_stage, to_stage=to_stage)
        stale = compute_stale_set(manifest, root, overrides, current_configs=current_configs)
        if not stale:
            LOG.info("Nothing to do: no stale stages for %s", workspace_id_str)
            self._refresh_timeline(root)
            return workspace_id_str, root
        self._run_stages(manifest, stale, start_at=stale[0], workspace_root=root)
        self._refresh_timeline(root)
        return workspace_id_str, root

    def run_oneshot(
        self,
        *,
        output_dir: Optional[Path] = None,
        skip_video: bool = False,
        force: bool = False,
        from_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One-shot ``prepare`` + ``generate`` for the legacy ``run`` UX.

        This is the single orchestration entry point the CLI ``run`` command
        (and the :class:`~src.pipeline.Pipeline` compatibility shim) call, so
        there is exactly one code path through the workspace orchestrator.

        Final artefacts are copied from ``<workspace>/output/`` into
        ``output_dir`` (preserving the ``final_audio.wav`` / ``final_video.mp4``
        contract that ``dub.sh`` reads from ``--output-dir``).
        """
        if force:
            # ``--no-cache`` semantics: wipe the workspace so every stage
            # rebuilds from scratch (mirrors the legacy Pipeline behaviour).
            wid, _, _ = self._id()
            root = self._resolve_workspace_root(wid)
            if root.exists():
                LOG.info("run_oneshot: force rebuild — clearing workspace %s", root)
                shutil.rmtree(root, ignore_errors=True)

        wid, root = self.prepare(force=force)
        to_stage = "mix" if skip_video else None
        self.generate(wid, from_stage=from_stage, to_stage=to_stage, force=force)

        result: Dict[str, Any] = {
            "workspace_id": wid,
            "workspace_root": str(root),
            "final_audio": str(root / "output" / "final_audio.wav"),
            "final_video": str(root / "output" / "final_video.mp4"),
        }

        # Deliver final artefacts into the requested output dir.
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("final_audio.wav", "final_video.mp4"):
                src = root / "output" / name
                if src.exists():
                    dst = output_dir / name
                    if src.resolve() != dst.resolve():
                        shutil.copy2(src, dst)
            result["output_dir"] = str(output_dir)
        return result

    def _refresh_consumer_inputs(
        self, manifest: Manifest, producer_output_path: str, workspace_root: Path
    ) -> None:
        """Update every StageRecord whose ``inputs`` reference ``producer_output_path``.

        Called after a stage's outputs have been promoted so that the
        downstream consumers see the new (real) hash instead of the
        empty placeholder recorded when the manifest was first
        created.
        """
        for consumer_name, consumer_rec in manifest.stages.items():
            new_inputs = []
            changed = False
            for ref in consumer_rec.inputs:
                if ref.path == producer_output_path:
                    fresh = self._maybe_artifact(workspace_root, producer_output_path)
                    if fresh is not None and fresh.sha256 != ref.sha256:
                        new_inputs.append(fresh)
                        changed = True
                        continue
                new_inputs.append(ref)
            if changed:
                consumer_rec.inputs = new_inputs

    # -- metadata.json ----------------------------------------------------

    def _backfill_manifest_from_disk(
        self, manifest: Manifest, root: Path
    ) -> None:
        """Populate the inputs/outputs of stages that have not yet been run.

        Called by :meth:`generate` so the invalidation DAG has a baseline
        for stages whose ``run`` was never invoked in this workspace
        (typically everything after ``translate`` on a fresh
        ``prepare()``).  Idempotent: only fills a stage's record if its
        inputs or outputs are currently empty.
        """
        for name in STAGE_ORDER:
            cls = STAGE_CLASSES.get(name)
            if cls is None:
                continue
            rec = manifest.stages.get(name)
            if rec is None:
                rec = StageRecord(name=name, status="pending")
                manifest.stages[name] = rec
            if not rec.inputs:
                rec.inputs = [
                    ref
                    for rel in getattr(cls, "inputs", [])
                    for ref in [self._maybe_artifact(root, rel)]
                    if ref is not None
                ]
            if not rec.outputs:
                rec.outputs = [
                    ref
                    for rel in getattr(cls, "outputs", [])
                    for ref in [self._maybe_artifact(root, rel)]
                    if ref is not None
                ]

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
        """Write ``<root>/metadata.json`` per spec §8 (idempotent).

        Rewritten when the workspace is new *or* the selected TTS backend
        changed, so metadata always reflects the current backend. The backend
        lives in the ``config`` block but NOT in the identity hash, so changing
        it keeps the same workspace while still being recorded.
        """
        meta_path = root / "metadata.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = None
            if (
                existing
                and existing.get("workspace_id") == _read_workspace_id(root)
                and (existing.get("config") or {}).get("tts_backend") == self.tts_backend
            ):
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
            # Backend recorded in config for visibility; excluded from the
            # identity hash (cfg_hash) so switching it keeps the workspace.
            "config": {**self._config(), "tts_backend": self.tts_backend},
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
        source_media = _find_source_media(workspace_root)
        context["input_path"] = str(source_media) if source_media else self.input_path
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
        context["manifest_path"] = str(workspace_root / "aligned_manifest.json")
        # ``generated_segments/manifest.json`` is the per-segment
        # output of the generate stage; the align stage overwrites
        # the context's ``manifest_path`` with the aligned manifest
        # when it runs (see :meth:`_run_single_stage`).  But the
        # reconstruct / mix / video stages need the aligned manifest
        # *even on a cold resume where align did not re-run*, so we
        # seed the context with the aligned-manifest path rather
        # than the generated one.  The align stage's return value
        # also points at the aligned manifest, so the seed and the
        # post-align state agree.
        context["aligned_dir"] = str(workspace_root / "aligned_segments")
        context["aligned_manifest"] = str(workspace_root / "aligned_manifest.json")
        context["reconstructed_path"] = str(workspace_root / "output" / "reconstructed_speech.wav")
        context["final_path"] = str(workspace_root / "output" / "final_audio.wav")
        context["final_video"] = str(workspace_root / "output" / "final_video.mp4")

        # Re-hydrate ``speaker_profiles`` from the persisted
        # ``speaker_profiles/`` tree so the generate stage has access
        # to voice profiles when reached on a resume where samples did
        # not re-run.  This is the only way the OmniVoice generate
        # stage can run outside of a single-session ``prepare()``
        # invocation.  If samples re-runs in this session its return
        # value will overwrite this dict (see ``_run_single_stage``).
        context["speaker_profiles"] = load_speaker_profiles_from_disk(workspace_root)

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
            tts_backend=self.tts_backend,
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
                result = stage.run(self.input_path, context._data)
            else:
                result = stage.run(context._data)
        except Exception as exc:  # noqa: BLE001 - we record and re-raise
            shutil.rmtree(staging, ignore_errors=True)
            failed = StageRecord(
                name=name,
                status="failed",
                started_at=started_at,
                finished_at=_now_iso(),
                duration_s=time.time() - started,
                config={
                    k: getattr(stage, k, None)
                    for k in getattr(stage_cls, "config_fields", [])
                },
                error=str(exc),
            )
            manifest.add_stage(name, failed)
            manifest.save(workspace_root / "manifest.json")
            LOG.error("Stage %s failed: %s", name, exc)
            raise

        # Rewrite any absolute paths in the return value that point into
        # the staging dir so they point into the workspace root instead.
        # The stage wrote its outputs to ``staging/...``; after promote
        # those files live in ``workspace_root/...`` but the stage's
        # return value (e.g. ``speaker_profiles[spk].reference``) still
        # carries the staging-dir absolute paths.  Without rewriting,
        # downstream stages would try to read from a directory that was
        # just removed by :func:`promote`.
        if isinstance(result, dict):
            staging_str = str(staging)
            workspace_str = str(workspace_root)
            result = _rewrite_staging_paths(result, staging_str, workspace_str)

        # Merge the stage's return value into the context so downstream
        # stages can read the upstream artefacts (e.g. the samples stage
        # returns ``{"speaker_profiles": ...}`` which the OmniVoice
        # generate stage needs from ``context["speaker_profiles"]``).
        # The legacy ``Pipeline.run`` does the same thing; the original
        # workspace orchestrator discarded the return value, which made
        # the generate stage raise ``"No speaker voice profiles
        # available"`` on every real workspace run.
        #
        # Special case: if a stage returns ``manifest_path``, that
        # always wins — the align stage, for example, returns the
        # path to ``aligned_manifest.json`` and downstream stages
        # (reconstruct, mix, video) need to read from there rather
        # than the generated-segments manifest the orchestrator
        # pre-populates.  Without this override, the reconstruct
        # stage would fail with "Manifest entries lack aligned_path".
        if isinstance(result, dict):
            for k, v in result.items():
                if k == "manifest_path" and v:
                    context._data[k] = v
                    continue
                # Don't clobber other well-known context keys the
                # orchestrator pre-populated from the workspace layout
                # (audio_path, speech_path, etc.) with stale per-stage
                # values.
                if k in context._data and k in _CONTEXT_KEYS:
                    continue
                context._data[k] = v

        # On success, promote staging into the workspace root.
        promote(staging, workspace_root)

        # Build the per-stage record: config, inputs, outputs (with hashes).
        config = {
            k: getattr(stage, k, None)
            for k in getattr(stage_cls, "config_fields", [])
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
        # Refresh the input records of *downstream* stages so they point
        # at the just-promoted files.  Without this, downstream
        # StageRecords keep the empty-sha256 placeholders the backfill
        # inserted at manifest-creation time, and the invalidation DAG
        # spuriously marks those downstream stages as stale from the
        # start.
        for out_ref in outputs:
            self._refresh_consumer_inputs(manifest, out_ref.path, workspace_root)
        manifest.save(workspace_root / "manifest.json")
        LOG.info("[ok] %s done in %.2fs", name, time.time() - started)

    @staticmethod
    def _maybe_artifact(workspace_root: Path, rel_path: str) -> Optional[ArtifactRef]:
        """Return an ``ArtifactRef`` for ``rel_path`` if it exists on disk.

        For files that do not exist yet (e.g. outputs of stages that
        have not run), return a *placeholder* ``ArtifactRef`` with
        ``sha256=""``.  The invalidation DAG treats a missing/empty
        recorded hash as a mismatch, so a downstream stage whose input
        has not been produced yet is correctly considered stale.  This
        is what enables the post-translate cascade to run end-to-end
        after a fresh ``prepare()``.
        """
        full = Path(workspace_root) / rel_path
        if not full.exists() or not full.is_file():
            return ArtifactRef(path=rel_path, sha256="", size_bytes=0)
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
            tts_backend=self.tts_backend,
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
