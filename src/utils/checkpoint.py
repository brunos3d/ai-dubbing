"""Checkpoint system enabling resumable pipeline runs."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageRecord:
    name: str
    status: str  # pending | running | done | failed | skipped
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_s: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageRecord":
        return cls(**data)


@dataclass
class Checkpoint:
    input_path: str
    source_language: str
    target_language: str
    workdir: str
    output_dir: str
    started_at: float
    updated_at: float
    stages: Dict[str, StageRecord] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def is_done(self, stage: str) -> bool:
        rec = self.stages.get(stage)
        return rec is not None and rec.status == "done"

    def mark_running(self, stage: str) -> None:
        now = time.time()
        rec = self.stages.get(stage)
        if rec is None:
            rec = StageRecord(name=stage, status="running", started_at=now)
        else:
            rec.status = "running"
            rec.started_at = now
            rec.error = None
        self.stages[stage] = rec
        self.updated_at = now
        self.save()

    def mark_done(
        self, stage: str, output_files: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        now = time.time()
        rec = self.stages.get(stage)
        if rec is None:
            rec = StageRecord(name=stage, status="done")
        rec.status = "done"
        rec.finished_at = now
        if rec.started_at is not None:
            rec.duration_s = now - rec.started_at
        if output_files is not None:
            rec.output_files = output_files
        if metadata is not None:
            rec.metadata = metadata
        self.stages[stage] = rec
        self.updated_at = now
        self.save()

    def mark_failed(self, stage: str, error: str) -> None:
        now = time.time()
        rec = self.stages.get(stage)
        if rec is None:
            rec = StageRecord(name=stage, status="failed")
        rec.status = "failed"
        rec.finished_at = now
        rec.error = error
        self.stages[stage] = rec
        self.updated_at = now
        self.save()

    def save(self) -> None:
        path = Path(self.workdir) / "checkpoint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        data = {
            "input_path": self.input_path,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "workdir": self.workdir,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "config": self.config,
        }
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(path)

    @classmethod
    def load(cls, workdir: str | Path) -> Optional["Checkpoint"]:
        path = Path(workdir) / "checkpoint.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        stages = {k: StageRecord.from_dict(v) for k, v in data.get("stages", {}).items()}
        return cls(
            input_path=data["input_path"],
            source_language=data["source_language"],
            target_language=data["target_language"],
            workdir=data["workdir"],
            output_dir=data["output_dir"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            stages=stages,
            config=data.get("config", {}),
        )

    @classmethod
    def create(
        cls,
        input_path: str,
        source_language: str,
        target_language: str,
        workdir: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> "Checkpoint":
        now = time.time()
        return cls(
            input_path=input_path,
            source_language=source_language,
            target_language=target_language,
            workdir=workdir,
            output_dir=output_dir,
            started_at=now,
            updated_at=now,
            config=config or {},
        )


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
