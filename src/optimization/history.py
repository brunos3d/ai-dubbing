"""Persistent, resumable optimization history.

Storage is deliberately tiny and media-free: one append-only JSONL file of
iteration records plus two small JSON snapshots (best config, loop state). No
audio or workspace data is ever duplicated here — the heavy artifacts live in
the single reused workspace and are overwritten in place each iteration, so
disk growth is bounded.

Layout under the run directory::

    <run_dir>/
        history.jsonl   # one IterationRecord per line (append-only)
        best.json       # the best record seen so far
        state.json      # next iteration index + RNG state (for resume)

A run can be stopped at any time and re-run later: :meth:`HistoryStore.resume`
reloads the history, the best record, and the loop state so the search picks up
exactly where it left off.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class IterationRecord:
    """One evaluated configuration and its result."""

    iteration: int
    score: Optional[float]
    parameters: Dict[str, Any]
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    status: str = "ok"  # "ok" | "failed"
    error: Optional[str] = None
    from_stage: Optional[str] = None
    duration_s: Optional[float] = None
    timestamp: Optional[str] = None
    source: str = "search"  # "baseline" | "search" | "random" | "resume"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IterationRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class HistoryStore:
    """Append-only history with best-config tracking and resume support."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.jsonl"
        self.best_path = self.run_dir / "best.json"
        self.state_path = self.run_dir / "state.json"

    # -- writing ----------------------------------------------------------

    def append(self, record: IterationRecord) -> None:
        """Append a record to the JSONL log and refresh ``best.json`` if better."""
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        best = self.best()
        if record.score is not None and (best is None or record.score > (best.score or -1e9)):
            _atomic_write(self.best_path, json.dumps(record.to_dict(), indent=2, ensure_ascii=False))

    def save_state(self, state: Dict[str, Any]) -> None:
        _atomic_write(self.state_path, json.dumps(state, indent=2, ensure_ascii=False))

    # -- reading ----------------------------------------------------------

    def load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def records(self) -> List[IterationRecord]:
        if not self.history_path.exists():
            return []
        out: List[IterationRecord] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(IterationRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def best(self) -> Optional[IterationRecord]:
        if not self.best_path.exists():
            return None
        try:
            return IterationRecord.from_dict(json.loads(self.best_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def next_iteration(self) -> int:
        recs = self.records()
        return (max((r.iteration for r in recs), default=-1) + 1)

    def resume(self) -> Dict[str, Any]:
        """Return everything needed to continue a run: records, best, state."""
        recs = self.records()
        return {
            "records": recs,
            "best": self.best(),
            "state": self.load_state(),
            "next_iteration": (max((r.iteration for r in recs), default=-1) + 1),
            "n_records": len(recs),
        }

    # -- reporting --------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        recs = self.records()
        ok = [r for r in recs if r.score is not None]
        best = self.best()
        scores = [r.score for r in ok]
        return {
            "n_iterations": len(recs),
            "n_ok": len(ok),
            "n_failed": len(recs) - len(ok),
            "best_score": (best.score if best else None),
            "best_iteration": (best.iteration if best else None),
            "best_parameters": (best.parameters if best else None),
            "score_progression": [round(s, 5) for s in scores] if scores else [],
        }
