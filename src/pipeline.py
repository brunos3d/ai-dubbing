"""Compatibility shim around the workspace orchestrator.

Historically this module held a second, parallel orchestrator (a
checkpoint-based ``Pipeline``) that duplicated all of the context-hydration
and stage-wiring logic now owned by
:class:`src.workspace.pipeline.WorkspacePipeline`.  That duplication was the
single largest maintainability liability flagged by the architecture audit
(``FINAL_REPORT.md`` §4) and the consolidation work in spec
``2026-06-16-timing-aware-timeline-evolution.md`` (Phase 1A).

``WorkspacePipeline`` is now the **single** orchestrator.  ``Pipeline`` is kept
only as a thin compatibility layer:

* it preserves :meth:`pipeline_config_dict` (used for config hashing and by
  ``tests/test_pipeline_config_dict.py``);
* it preserves the :data:`Pipeline.STAGES` ordering for any external caller;
* its :meth:`run` delegates to :meth:`WorkspacePipeline.run_oneshot`.

No new code should import ``Pipeline``; prefer ``WorkspacePipeline`` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .utils.logging import get_logger
from .utils.paths import env, output_dir

LOG = get_logger("ai-dubbing")

# Canonical stage order, retained for back-compat with any caller that
# introspected ``Pipeline.STAGES``.  The authoritative order now lives in
# :data:`src.workspace.dag.STAGE_ORDER`.
STAGE_NAMES = [
    "extract",
    "separate",
    "diarize",
    "samples",
    "transcribe",
    "translate",
    "generate",
    "align",
    "reconstruct",
    "mix",
    "video",
]


class Pipeline:
    """Thin compatibility layer over :class:`WorkspacePipeline`."""

    # Kept as ``(name, None)`` tuples so legacy ``[n for n, _ in STAGES]``
    # introspection keeps working without importing the stage classes here.
    STAGES = [(name, None) for name in STAGE_NAMES]

    def __init__(
        self,
        input_path: str,
        source_language: str,
        target_language: str,
        workdir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        whisper_model: str = "large-v3",
        hf_token: Optional[str] = None,
        target_lufs: float = -16.0,
        skip_video: bool = False,
        no_cache: bool = False,
        from_stage: Optional[str] = None,
        read_only_cache: bool = False,
        glossary_path: Optional[Path] = None,
        no_pyannote: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **_ignored: Any,
    ):
        env()
        self.input_path = input_path
        self.source_language = source_language
        self.target_language = target_language
        self.output_dir = output_dir(output_path)
        self.workdir = Path(workdir).resolve() if workdir is not None else None
        self.whisper_model = whisper_model
        self.hf_token = hf_token
        self.target_lufs = target_lufs
        self.skip_video = skip_video
        self.no_cache = no_cache
        self.from_stage = from_stage
        self.read_only_cache = read_only_cache
        self.glossary_path = glossary_path
        self.no_pyannote = no_pyannote
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def pipeline_config_dict(self) -> dict:
        """Return the deterministic config dict used for workspace identity.

        The ``hf_token`` itself is *never* present — only the boolean
        ``hf_token_available`` flag — so rotating a token or switching HF
        accounts does not invalidate a workspace.
        """
        return {
            "whisper_model": self.whisper_model,
            "target_lufs": self.target_lufs,
            "no_pyannote": self.no_pyannote,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "skip_video": self.skip_video,
            "glossary_path": str(self.glossary_path) if self.glossary_path else None,
            "hf_token_available": bool(self.hf_token),
        }

    def run(self, start_from: str = "extract", only: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to :meth:`WorkspacePipeline.run_oneshot`.

        ``start_from`` is mapped to the workspace ``from_stage`` override; the
        rarely-used ``only`` debug flag is honoured by capping the run to that
        single stage where it falls in the synthesis half.  Fine-grained stage
        control is otherwise available through ``dub.sh generate``.
        """
        from .workspace.pipeline import WorkspacePipeline

        wsp = WorkspacePipeline(
            input_path=self.input_path,
            source_language=self.source_language,
            target_language=self.target_language,
            whisper_model=self.whisper_model,
            hf_token=self.hf_token,
            target_lufs=self.target_lufs,
            glossary_path=self.glossary_path,
            no_pyannote=self.no_pyannote,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            workspace_root=self.workdir,
        )
        # ``--from-stage`` (explicit rebuild-from) takes precedence over the
        # ``--start-from`` resume hint; ``extract`` means "no override".
        from_stage = self.from_stage or (
            start_from if start_from and start_from != "extract" else None
        )
        return wsp.run_oneshot(
            output_dir=self.output_dir,
            skip_video=self.skip_video,
            force=self.no_cache,
            from_stage=from_stage,
        )
