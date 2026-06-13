# ai-dubbing

Local-first multilingual dubbing pipeline that turns a video (or audio) file in
one language into a translated, voice-cloned dub in another language while
preserving the original speakers' voices, pacing, ambience, music and effects.

## Quick start

```bash
./dub.sh input/video.mp4 pt en                  # pt -> en; default: full video
./dub.sh input/video.mp4 en pt-BR --audio-only  # only the dubbed audio
./dub.sh input/video.mp4 pt en --glossary g.json # preserve entities
./dub.sh cache clear                            # clear all artifacts
./dub.sh glossary template                      # create a glossary file
```

By default the dub script delivers the full dubbed video next to the input file. Use flags to control the output and behavior:

| Flag | Effect |
|---|---|
| `--audio-only` | produce only `final_audio.wav` (no video remux) |
| `--glossary PATH` | use a JSON glossary to prevent names/brands from being translated |
| `--adapt-mode MODE`| set the persona (YouTube Narrator, Documentary, Podcast, Casual, News) |
| `--no-cache` | force a complete rebuild by ignoring existing cache |
| `--from-stage NAME` | reuse cache up to NAME, then rebuild everything after |

## Pipeline

1. **Audio Extraction** — FFmpeg
2. **Source Separation** — Demucs (htdemucs)
3. **Speaker Diarization** — pyannote.audio
4. **Reference Sample Extraction** — Quality-based selection (8–12 s)
5. **Speech Recognition** — faster-whisper (large-v3) with low-confidence re-verification
6. **Translation** — NLLB-200 (facebook/nllb-200-distilled-600M)
7. **Speech Adaptation** — SmolLM2 (HuggingFaceTB/SmolLM2-1.7B-Instruct)
8. **Voice Generation** — OmniVoice (k2-fsa/OmniVoice)
9. **Duration Alignment** — FFmpeg `atempo` (librosa fallback)
10. **Timeline Reconstruction** — overlay + background mix
11. **Final Mix** — FFmpeg with EBU R128 loudness normalization
12. **Optional Video** — remux dubbed audio onto the original video

## Professional Dubbing Tips

### Named Entity Preservation
To prevent characters or brands from being translated (e.g., "Ei Nerd" becoming "Hey Nerd"), use a glossary:
1. Generate a template: `./dub.sh glossary template --output entities.json`
2. Add your terms to the JSON file.
3. Run with: `./dub.sh input.mp4 pt en --glossary entities.json`

### Tone and Personality
Use `--adapt-mode` to match the style of the original content. The default "YouTube Narrator" is high-energy. For educational content, "Documentary" provides a calmer delivery.

## Requirements

- Python 3.12
- FFmpeg on PATH
- An NVIDIA GPU with at least ~3 GB of free VRAM (8 GB recommended)
- Hugging Face token (`HF_TOKEN`) for high-quality pyannote diarization
  (optional — a VAD-based fallback is used otherwise)
