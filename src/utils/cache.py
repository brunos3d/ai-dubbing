"""Cache management for the dubbing pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logging import get_logger

LOG = get_logger("ai-dubbing.cache")


def media_hash(file_path: str | Path) -> str:
    """Compute SHA256 hash of the input file content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def project_hash() -> str:
    """Compute a hash representing the current project state.

    Uses the git commit hash if available, falling back to hashing
    all source files in the project.
    """
    root = Path(__file__).resolve().parent.parent.parent
    
    # Attempt to get git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            cwd=str(root)
        ).strip()
        
        # Check if dirty - if so, we should probably append something
        # to ensure local changes invalidate the cache.
        is_dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
            cwd=str(root)
        ) != 0
        
        if not is_dirty:
            return f"git-{git_hash}"
    except Exception:
        pass

    # Fallback or Dirty: Hash source files
    h = hashlib.sha256()
    # Relevant files that affect pipeline behavior
    files = []
    for p in (root / "src").glob("**/*.py"):
        files.append(p)
    files.append(root / "main.py")
    if (root / "pyproject.toml").exists():
        files.append(root / "pyproject.toml")
    
    for p in sorted(files):
        if p.exists():
            # Hash path + mtime + size for speed, or full content for accuracy.
            # Content is safer for "implementation changes".
            h.update(p.read_bytes())
    
    return f"src-{h.hexdigest()[:16]}"


class CacheManager:
    """Manages the lifecycle of temporary pipeline artifacts."""

    def __init__(self, root: Optional[Path] = None):
        if root is None:
            self.root = Path(tempfile.gettempdir()) / "ai-dubbing"
        else:
            self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, media_path: str | Path) -> str:
        """Generate a deterministic cache key for the given input."""
        m_hash = media_hash(media_path)
        p_hash = project_hash()
        combined = f"{m_hash}:{p_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_working_dir(self, cache_key: str) -> Path:
        """Get the path to the working directory for a given cache key."""
        d = self.root / cache_key
        return d

    def init_metadata(self, cache_key: str, media_path: str | Path, config: Dict[str, Any]) -> None:
        """Initialize metadata for a new cache entry."""
        wdir = self.get_working_dir(cache_key)
        wdir.mkdir(parents=True, exist_ok=True)
        meta_path = wdir / "metadata.json"
        
        data = {
            "cache_key": cache_key,
            "media_path": str(Path(media_path).resolve()),
            "media_hash": media_hash(media_path),
            "project_hash": project_hash(),
            "created_at": time.time(),
            "updated_at": time.time(),
            "config": config,
        }
        meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def update_metadata(self, cache_key: str) -> None:
        """Update the 'updated_at' timestamp for a cache entry."""
        meta_path = self.get_working_dir(cache_key) / "metadata.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                data["updated_at"] = time.time()
                meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            except Exception:
                pass

    def get_metadata(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a cache entry."""
        meta_path = self.get_working_dir(cache_key) / "metadata.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all valid cache entries."""
        entries = []
        if not self.root.exists():
            return []
            
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            meta = self.get_metadata(d.name)
            if meta:
                # Add size info
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                meta["size_bytes"] = size
                entries.append(meta)
        return sorted(entries, key=lambda x: x["updated_at"], reverse=True)

    def clear(self) -> int:
        """Remove all cache entries. Returns number of entries removed."""
        count = 0
        if not self.root.exists():
            return 0
        for d in self.root.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
                count += 1
            else:
                d.unlink()
        return count

    def prune(self, max_age_days: float = 7.0) -> int:
        """Remove stale entries older than max_age_days."""
        count = 0
        now = time.time()
        max_age_s = max_age_days * 86400
        for entry in self.list_entries():
            if now - entry["updated_at"] > max_age_s:
                wdir = self.get_working_dir(entry["cache_key"])
                if wdir.exists():
                    shutil.rmtree(wdir)
                    count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics about the cache."""
        entries = self.list_entries()
        total_size = sum(e["size_bytes"] for e in entries)
        if not entries:
            return {
                "root": str(self.root),
                "count": 0,
                "total_size_bytes": 0,
                "oldest": None,
                "newest": None,
            }
        
        return {
            "root": str(self.root),
            "count": len(entries),
            "total_size_bytes": total_size,
            "oldest": min(e["updated_at"] for e in entries),
            "newest": max(e["updated_at"] for e in entries),
        }
