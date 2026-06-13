# ai-dubbing

Local-first multilingual dubbing pipeline that turns a video (or audio) file in
one language into a translated, voice-cloned dub in another language while
preserving the original speakers' voices, pacing, ambience, music and effects.

## Quick start

```bash
./dub.sh input/video.mp4 pt en                  # pt -> en; default: full video
./dub.sh input/video.mp4 en pt-BR --audio-only  # only the dubbed audio
./dub.sh input/song.mp3 en es --emit=audio      # explicit audio-only
python main.py -i input.mp4 -s en -t es         # direct CLI access
```

By default the dub script delivers the full dubbed video at
`output/final_video.mp4` (and the standalone `output/final_audio.wav` as a
byproduct). Use a flag to take only the audio:

| Flag | Effect |
|---|---|
| *(default)* | produce both `final_audio.wav` and `final_video.mp4` |
| `--audio-only` | produce only `final_audio.wav` (no video remux) |
| `--emit=audio` | same as `--audio-only` |
| `--emit=video` | produce only the video (audio file still written) |
| `--emit=both` | same as default |
| `-- --whisper-model=medium` | forward any `main.py` flag through the wrapper |

## Pipeline

1. **Audio Extraction** — FFmpeg
2. **Source Separation** — Demucs (htdemucs)
3. **Speaker Diarization** — pyannote.audio (with VAD + MFCC clustering fallback)
4. **Reference Sample Extraction** — clean 5–10 s clips per speaker
5. **Speech Recognition** — faster-whisper (large-v3)
6. **Translation** — Google / MyMemory via deep-translator
7. **Voice Generation** — OmniVoice (k2-fsa/OmniVoice)
8. **Duration Alignment** — FFmpeg `atempo` (librosa fallback)
9. **Timeline Reconstruction** — overlay + background mix
10. **Final Mix** — FFmpeg with EBU R128 loudness normalization
11. **Optional Video** — remux dubbed audio onto the original video

## Requirements

- Python 3.12
- FFmpeg on PATH
- An NVIDIA GPU with at least ~3 GB of free VRAM (8 GB recommended)
- Hugging Face token (`HF_TOKEN`) for high-quality pyannote diarization
  (optional — a VAD-based fallback is used otherwise)
