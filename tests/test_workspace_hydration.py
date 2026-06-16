import json
import pytest
from pathlib import Path
from src.workspace.pipeline import WorkspacePipeline
from src.workspace.manifest import Manifest

def test_generate_hydrates_from_metadata(tmp_path, monkeypatch):
    """Verify that generate() hydrates settings from metadata.json."""
    # Mock workspaces root
    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    
    # Create a dummy workspace
    wid = "test-20260614-abc12345"
    root = tmp_path / wid
    root.mkdir(parents=True)
    
    # Create metadata.json with non-default settings
    metadata = {
        "source": {
            "media_path": "/real/path/video.mp4",
            "source_language": "en",
            "target_language": "pt-BR"
        },
        "config": {
            "whisper_model": "medium",
            "target_lufs": -14.0,
            "no_pyannote": True,
            "min_speakers": 2,
            "max_speakers": 5
        }
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    
    # Create a minimal manifest.json
    manifest = Manifest.create(
        workspace_id=wid,
        pipeline_version="1.0.0",
        git_commit="head"
    )
    manifest.save(root / "manifest.json")
    
    # Instantiate pipeline with placeholders (as the 'generate' CLI does)
    wsp = WorkspacePipeline(
        input_path="/dev/null",
        source_language="?",
        target_language="?"
    )
    
    # Initially it has defaults
    assert wsp.whisper_model == "large-v3"
    assert wsp.target_lufs == -16.0
    
    # Mock _run_stages to avoid full execution
    monkeypatch.setattr(wsp, "_run_stages", lambda *args, **kwargs: None)
    
    # Run generate
    wsp.generate(wid)
    
    # Verify hydration
    assert wsp.input_path == "/real/path/video.mp4"
    assert wsp.source_language == "en"
    assert wsp.target_language == "pt-BR"
    assert wsp.whisper_model == "medium"
    assert wsp.target_lufs == -14.0
    assert wsp.no_pyannote is True
    assert wsp.min_speakers == 2
    assert wsp.max_speakers == 5
