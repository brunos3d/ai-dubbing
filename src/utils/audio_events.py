"""Audio event classification for reaction and ambience detection.

This module provides tools to classify non-speech audio events into categories:
- Reactions: laughter, applause, gasps, coughs, cheers
- Ambience: room tone, crowd noise, environmental sounds

Used by the multi-layer separation system to preserve audience reactions
and environmental context during reconstruction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

LOG = get_logger("ai-dubbing.audio_events")


@dataclass
class AudioEvent:
    """A classified audio event with timing and category."""
    event_type: str  # "laughter", "applause", "ambience", etc.
    start: float
    end: float
    confidence: float = 1.0
    intensity: float = 1.0  # 0-1 normalized energy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "confidence": round(float(self.confidence), 3),
            "intensity": round(float(self.intensity), 3),
        }


def _frame_energy(audio: np.ndarray, sr: int, frame_ms: float = 50.0) -> np.ndarray:
    """Compute per-frame RMS energy."""
    win = max(1, int(frame_ms / 1000.0 * sr))
    hop = win // 2
    if audio.shape[0] < win:
        return np.array([np.sqrt(np.mean(audio ** 2))], dtype="float32")
    n = 1 + (audio.shape[0] - win) // hop
    energy = np.empty(n, dtype="float32")
    for i in range(n):
        seg = audio[i * hop:i * hop + win]
        energy[i] = np.sqrt(np.mean(seg ** 2))
    return energy


def _spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid - higher values indicate brighter sounds."""
    try:
        import librosa
        centroid = librosa.feature.spectral_centroid(y=audio.astype("float32"), sr=sr)
        return float(np.mean(centroid))
    except Exception:
        return 0.0


def _spectral_bandwidth(audio: np.ndarray, sr: int) -> float:
    """Compute spectral bandwidth - wider bandwidth indicates more complex sounds."""
    try:
        import librosa
        bandwidth = librosa.feature.spectral_bandwidth(y=audio.astype("float32"), sr=sr)
        return float(np.mean(bandwidth))
    except Exception:
        return 0.0


def _zero_crossing_rate(audio: np.ndarray, sr: int) -> float:
    """Compute zero crossing rate - higher for percussive/noisy sounds."""
    try:
        import librosa
        zcr = librosa.feature.zero_crossing_rate(y=audio.astype("float32"))
        return float(np.mean(zcr))
    except Exception:
        return 0.0


def detect_reactions(
    audio: np.ndarray,
    sr: int,
    speech_segments: List[Dict[str, Any]],
    min_duration: float = 0.3,
    energy_threshold_ratio: float = 0.3,
) -> List[AudioEvent]:
    """Detect audience reactions (laughter, applause) in non-speech regions.

    Uses energy-based detection with spectral features to classify events.
    Reactions typically have:
    - High energy bursts
    - Specific spectral characteristics (laughter: 200-2000Hz, applause: broadband)
    - Occur in gaps between speech segments
    """
    if audio.ndim > 1:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    # Build non-speech mask
    n_samples = mono.shape[0]
    is_speech = np.zeros(n_samples, dtype=bool)
    for seg in speech_segments:
        s = max(0, int(seg["start"] * sr))
        e = min(n_samples, int(seg["end"] * sr))
        is_speech[s:e] = True

    # Compute frame energy
    frame_ms = 50.0
    energy = _frame_energy(mono, sr, frame_ms)
    hop_samples = int(frame_ms / 1000.0 * sr) // 2

    if energy.size == 0:
        return []

    # Find high-energy regions in non-speech areas
    peak_energy = float(np.percentile(energy, 95))
    threshold = energy_threshold_ratio * peak_energy

    events: List[AudioEvent] = []
    in_event = False
    event_start = 0
    event_energy_sum = 0.0
    event_frames = 0

    for i, e in enumerate(energy):
        frame_start_sample = i * hop_samples
        frame_mid_sample = min(frame_start_sample + hop_samples, n_samples - 1)

        # Skip if this frame is in a speech region
        if is_speech[frame_mid_sample]:
            if in_event:
                # End the event
                event_end = i * hop_samples / sr
                event_duration = event_end - event_start
                if event_duration >= min_duration:
                    avg_energy = event_energy_sum / max(1, event_frames)
                    intensity = min(1.0, avg_energy / max(1e-6, peak_energy))
                    # Classify the event
                    event_audio = mono[int(event_start * sr):int(event_end * sr)]
                    event_type = _classify_reaction(event_audio, sr)
                    events.append(AudioEvent(
                        event_type=event_type,
                        start=event_start,
                        end=event_end,
                        confidence=0.7,  # Energy-based detection
                        intensity=intensity,
                    ))
                in_event = False
                event_energy_sum = 0.0
                event_frames = 0
            continue

        if e > threshold:
            if not in_event:
                event_start = i * hop_samples / sr
                in_event = True
            event_energy_sum += e
            event_frames += 1
        else:
            if in_event:
                event_end = i * hop_samples / sr
                event_duration = event_end - event_start
                if event_duration >= min_duration:
                    avg_energy = event_energy_sum / max(1, event_frames)
                    intensity = min(1.0, avg_energy / max(1e-6, peak_energy))
                    event_audio = mono[int(event_start * sr):int(event_end * sr)]
                    event_type = _classify_reaction(event_audio, sr)
                    events.append(AudioEvent(
                        event_type=event_type,
                        start=event_start,
                        end=event_end,
                        confidence=0.7,
                        intensity=intensity,
                    ))
                in_event = False
                event_energy_sum = 0.0
                event_frames = 0

    # Handle event at end
    if in_event:
        event_end = len(energy) * hop_samples / sr
        event_duration = event_end - event_start
        if event_duration >= min_duration:
            avg_energy = event_energy_sum / max(1, event_frames)
            intensity = min(1.0, avg_energy / max(1e-6, peak_energy))
            event_audio = mono[int(event_start * sr):int(event_end * sr)]
            event_type = _classify_reaction(event_audio, sr)
            events.append(AudioEvent(
                event_type=event_type,
                start=event_start,
                end=event_end,
                confidence=0.7,
                intensity=intensity,
            ))

    return events


def _classify_reaction(audio: np.ndarray, sr: int) -> str:
    """Classify a reaction event by its spectral characteristics.

    Heuristics:
    - Laughter: periodic, mid-frequency (200-2000Hz), rhythmic
    - Applause: broadband, noisy, high zero-crossing rate
    - Cheers: sustained, broadband, high energy
    - Generic reaction: fallback
    """
    if audio.size < int(0.1 * sr):
        return "reaction"

    centroid = _spectral_centroid(audio, sr)
    bandwidth = _spectral_bandwidth(audio, sr)
    zcr = _zero_crossing_rate(audio, sr)

    # Normalize features (rough heuristics based on typical values)
    # Spectral centroid: laughter ~500-1500Hz, applause ~2000-4000Hz
    # ZCR: applause/cheers have higher ZCR than laughter

    if centroid > 2500 and zcr > 0.1:
        return "applause"
    elif centroid > 2000 and bandwidth > 2000:
        return "cheers"
    elif 300 < centroid < 2000 and zcr < 0.15:
        return "laughter"
    else:
        return "reaction"


def detect_ambience_regions(
    audio: np.ndarray,
    sr: int,
    speech_segments: List[Dict[str, Any]],
    reaction_events: List[AudioEvent],
    min_duration: float = 0.5,
) -> List[AudioEvent]:
    """Detect continuous ambience regions (room tone, environmental sounds).

    Ambience is what remains after removing speech and reactions.
    These are typically low-energy, continuous background sounds.
    """
    if audio.ndim > 1:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    n_samples = mono.shape[0]

    # Build mask of "active" regions (speech + reactions)
    is_active = np.zeros(n_samples, dtype=bool)

    for seg in speech_segments:
        s = max(0, int(seg["start"] * sr))
        e = min(n_samples, int(seg["end"] * sr))
        is_active[s:e] = True

    for event in reaction_events:
        s = max(0, int(event.start * sr))
        e = min(n_samples, int(event.end * sr))
        is_active[s:e] = True

    # Find continuous inactive regions
    ambience_regions: List[AudioEvent] = []
    in_region = False
    region_start = 0

    for i in range(n_samples):
        if not is_active[i]:
            if not in_region:
                region_start = i
                in_region = True
        else:
            if in_region:
                region_end = i
                duration = (region_end - region_start) / sr
                if duration >= min_duration:
                    region_audio = mono[region_start:region_end]
                    energy = float(np.sqrt(np.mean(region_audio ** 2)))
                    ambience_regions.append(AudioEvent(
                        event_type="ambience",
                        start=region_start / sr,
                        end=region_end / sr,
                        confidence=0.9,
                        intensity=min(1.0, energy * 10),  # Normalize
                    ))
                in_region = False

    # Handle region at end
    if in_region:
        region_end = n_samples
        duration = (region_end - region_start) / sr
        if duration >= min_duration:
            region_audio = mono[region_start:region_end]
            energy = float(np.sqrt(np.mean(region_audio ** 2)))
            ambience_regions.append(AudioEvent(
                event_type="ambience",
                start=region_start / sr,
                end=region_end / sr,
                confidence=0.9,
                intensity=min(1.0, energy * 10),
            ))

    return ambience_regions


def classify_audio_layers(
    audio: np.ndarray,
    sr: int,
    speech_segments: List[Dict[str, Any]],
) -> Dict[str, List[AudioEvent]]:
    """Classify audio into three layers: speech, reactions, ambience.

    Returns a dict with keys:
    - "speech": the input speech segments (passed through)
    - "reactions": detected audience reactions
    - "ambience": detected ambience regions
    """
    LOG.info(f"Classifying audio layers ({len(speech_segments)} speech segments)")

    reactions = detect_reactions(audio, sr, speech_segments)
    LOG.info(f"Detected {len(reactions)} reaction events")

    ambience = detect_ambience_regions(audio, sr, speech_segments, reactions)
    LOG.info(f"Detected {len(ambience)} ambience regions")

    return {
        "speech": [AudioEvent("speech", s["start"], s["end"]) for s in speech_segments],
        "reactions": reactions,
        "ambience": ambience,
    }


def save_event_manifest(events: Dict[str, List[AudioEvent]], path: Path) -> None:
    """Save classified events to a JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        category: [e.to_dict() for e in event_list]
        for category, event_list in events.items()
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_event_manifest(path: Path) -> Dict[str, List[AudioEvent]]:
    """Load classified events from a JSON manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        category: [AudioEvent(**e) for e in event_list]
        for category, event_list in data.items()
    }
