"""Workspace manifest: tracks per-stage inputs/outputs with SHA-256 hashes.

The manifest is a small JSON document persisted at ``<workspace>/manifest.json``
that lets the invalidation DAG (see :mod:`src.workspace.dag`) decide which
stages need to re-run after a user edit. It is the durable record of "what was
produced from what" and is the only file the orchestrator reads to resume a
workspace.

Key invariants:

* **Atomic writes** — the manifest is written via tmp-file + rename so a crash
  mid-write never leaves a half-written file on disk.
* **Schema-versioned** — :data:`SCHEMA_VERSION` lets future migrations fail
  loudly instead of silently misinterpreting an old workspace.
* **Hash-addressed** — every input and output carries a SHA-256 of its
  contents, enabling the invalidation algorithm in spec §10.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION: int = 1


class ManifestError(ValueError):
    """Raised when a manifest file exists but cannot be parsed or is invalid."""


@dataclass
class ArtifactRef:
    """A reference to a file in the workspace: its path and a content hash."""

    path: str
    sha256: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArtifactRef":
        return cls(
            path=str(d["path"]),
            sha256=str(d["sha256"]),
            size_bytes=int(d.get("size_bytes", 0)),
        )


@dataclass
class StageRecord:
    """The durable record of a single stage run."""

    name: str
    status: str = "pending"
    config: dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    inputs: list[ArtifactRef] = field(default_factory=list)
    outputs: list[ArtifactRef] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "inputs": [a.to_dict() for a in self.inputs],
            "outputs": [a.to_dict() for a in self.outputs],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageRecord":
        return cls(
            name=str(d["name"]),
            status=str(d.get("status", "pending")),
            config=dict(d.get("config", {})),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            duration_s=d.get("duration_s"),
            inputs=[ArtifactRef.from_dict(x) for x in d.get("inputs", [])],
            outputs=[ArtifactRef.from_dict(x) for x in d.get("outputs", [])],
            error=d.get("error"),
        )


@dataclass
class Manifest:
    """The workspace manifest: schema, identity, editable/derived paths, stages."""

    schema_version: int
    workspace_id: str
    pipeline_version: str
    git_commit: str
    editable_paths: list[str] = field(default_factory=list)
    derived_paths: list[str] = field(default_factory=list)
    stages: dict[str, StageRecord] = field(default_factory=dict)

    @classmethod
    def create(
        cls, workspace_id: str, pipeline_version: str, git_commit: str
    ) -> "Manifest":
        return cls(
            schema_version=SCHEMA_VERSION,
            workspace_id=workspace_id,
            pipeline_version=pipeline_version,
            git_commit=git_commit,
        )

    def add_stage(self, name: str, record: StageRecord) -> None:
        self.stages[name] = record

    def add_editable_path(self, path: str) -> None:
        if path not in self.editable_paths:
            self.editable_paths.append(path)

    def add_derived_path(self, path: str) -> None:
        if path not in self.derived_paths:
            self.derived_paths.append(path)

    def find_producer(self, path: str) -> Optional[str]:
        for stage_name, stage in self.stages.items():
            for out in stage.outputs:
                if out.path == path:
                    return stage_name
        return None

    def get_input(self, stage_name: str, path: str) -> Optional[ArtifactRef]:
        stage = self.stages.get(stage_name)
        if stage is None:
            return None
        for ref in stage.inputs:
            if ref.path == path:
                return ref
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workspace_id": self.workspace_id,
            "pipeline_version": self.pipeline_version,
            "git_commit": self.git_commit,
            "editable_paths": list(self.editable_paths),
            "derived_paths": list(self.derived_paths),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Optional["Manifest"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest at {path} is not valid JSON: {exc}") from exc
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            workspace_id=str(data["workspace_id"]),
            pipeline_version=str(data["pipeline_version"]),
            git_commit=str(data["git_commit"]),
            editable_paths=list(data.get("editable_paths", [])),
            derived_paths=list(data.get("derived_paths", [])),
            stages={
                k: StageRecord.from_dict(v) for k, v in data.get("stages", {}).items()
            },
        )
