import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.stages.transcribe import _reverify_low_confidence
from src.stages.samples import _transcribe_reference

def test_whisper_reuse_in_transcribe():
    """Verify that _reverify_low_confidence reuses the passed model."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], None)
    
    segment = {"text": "hello", "start": 0, "end": 1, "avg_logprob": -1.5}
    
    with patch("faster_whisper.WhisperModel") as MockWhisper:
        # Pass the mock model
        _reverify_low_confidence(
            Path("dummy.wav"), segment, "tiny", "en", "cpu", "int8", 
            threshold=-0.5, model=mock_model
        )
        
        # WhisperModel should NOT have been instantiated
        MockWhisper.assert_not_called()
        # mock_model.transcribe SHOULD have been called
        mock_model.transcribe.assert_called_once()

def test_whisper_reuse_in_samples():
    """Verify that _transcribe_reference reuses the passed model."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], None)
    
    with patch("faster_whisper.WhisperModel") as MockWhisper:
        # Pass the mock model
        _transcribe_reference(Path("dummy.wav"), "en", model=mock_model)
        
        # WhisperModel should NOT have been instantiated
        MockWhisper.assert_not_called()
        # mock_model.transcribe SHOULD have been called
        mock_model.transcribe.assert_called_once()
