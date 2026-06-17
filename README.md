# ai-dubbing

Local-first multilingual dubbing pipeline that turns a video (or audio) file in
one language into a translated, voice-cloned dub in another language while
preserving the original speakers' voices, pacing, ambience, music and effects.

## Quick start

```bash
./dub.sh input/video.mp4 pt en                  # pt -> en; default: full video
./dub.sh input/video.mp4 en pt-BR --audio-only  # only the dubbed audio
./dub.sh input/video.mp4 pt en --glossary g.json # preserve entities
./dub.sh prepare input/video.mp4 pt en          # create a workspace, stop at translate
./dub.sh generate <workspace-id>                # resume the workspace, produce output
./dub.sh cache clear                            # clear all artifacts
./dub.sh glossary template                      # create a glossary file
```

By default the dub script delivers the full dubbed video next to the input file. Use flags to control the output and behavior:

| Flag | Effect |
|---|---|
| `--audio-only` | produce only `final_audio.wav` (no video remux) |
| `--glossary PATH` | use a JSON glossary to prevent names/brands from being translated |
| `--no-cache` | force a complete rebuild by ignoring existing cache |
| `--from-stage NAME` | reuse cache up to NAME, then rebuild everything after |

## Pipeline

1. **Audio Extraction** — FFmpeg
2. **Source Separation** — Demucs (htdemucs)
3. **Speaker Diarization** — pyannote.audio
4. **Reference Sample Extraction** — Quality-based selection (8–12 s)
5. **Speech Recognition** — faster-whisper (large-v3) with low-confidence re-verification
6. **Timing-aware Translation** — `deep-translator` (Google → MyMemory fallback) through a pluggable `TranslationBackend`. Multiple candidates are generated per line; each is scored on **estimated spoken duration**, rate naturalness, fidelity, and glossary compliance, and the best-fitting candidate is chosen — so a translation that fits the slot is preferred over stretching audio. See [Timing-aware dubbing](docs/timing-aware-dubbing.md).
7. **Voice Generation** — OmniVoice (k2-fsa/OmniVoice), conditioned on a source-derived **prosody `speed`**, with **segment-level re-rendering** (only changed segments are re-synthesized).
8. **Duration Alignment** — FFmpeg `atempo` (librosa fallback), now the *single* duration-correction authority and rarely triggered thanks to timing-aware translation.
9. **Timeline Reconstruction** — overlay + background mix, with per-segment **prosody gain** preserving emphasis.
10. **Final Mix** — FFmpeg with EBU R128 loudness normalization, **dynamic ducking** (sidechain) and **light room matching**.
11. **Optional Video** — remux dubbed audio onto the original video

- **Two-phase workflow** — `prepare` runs stages 1–6 (analysis); `generate` runs stages 7–11 (synthesis). The split lets you inspect, edit, and resume a workspace without re-running the heavy analysis stages. `WorkspacePipeline` is the single orchestrator (the legacy `Pipeline` is a thin compatibility shim). See [Workspaces](docs/workspaces.md).
- **Canonical Timeline** — every run writes a derived `timeline.json` (per-segment speaker/text/timing/prosody/generation/review). See [Timing-aware dubbing](docs/timing-aware-dubbing.md).

## Workspaces

A **workspace** is a persistent directory at
`~/.local/share/ai-dubbing/workspaces/<slug>-<YYYYMMDD>-<hash8>/` that
captures every intermediate artefact the pipeline produces. Edit a
translation, swap a speaker sample, or add a glossary entry, then
`./dub.sh generate <workspace-id>` re-runs only the affected stages.

| Command | What it does |
|---|---|
| `./dub.sh prepare <input> <src> <tgt>` | run analysis (extract → translate), stop |
| `./dub.sh generate <workspace-id>` | run synthesis (TTS → final mix/video), DAG-driven |
| `./dub.sh workspace list` | list all workspaces |
| `./dub.sh workspace inspect <id>` | show the manifest stage table |
| `./dub.sh workspace show <id> [path]` | print a subdir path (or all) |
| `./dub.sh workspace validate <id>` | run every per-artifact validator |
| `./dub.sh workspace clean <id> [--keep-outputs] [--yes]` | delete a workspace |

The legacy one-shot `./dub.sh input.mp4 pt en` is unchanged: it runs
`prepare` + `generate` internally and prints the workspace ID at the end.

See [docs/workspaces.md](docs/workspaces.md) for the full guide — the
two-phase workflow, the five user-editable artefact paths, the editable /
non-editable / derived artefact classes, common edit-then-resume workflows,
and the `AI_DUBBING_WORKSPACES_ROOT` / `XDG_DATA_HOME` environment
variables.

## Self-optimization

`optimize.sh` automatically searches for better pipeline parameters by
repeatedly dubbing the same clip in a **same-language benchmark** (e.g.
English → English) and scoring how closely the output matches the source:

```bash
./optimize.sh run    -m input/clip.mp4 -l en        # infinite, resumable (Ctrl-C to stop)
./optimize.sh run    -m input/clip.mp4 -l en -n 50  # bounded
./optimize.sh status -m input/clip.mp4 -l en        # progress / best so far
./optimize.sh promote <path/to/best.json>           # update global pipeline defaults
```

You can also run a one-shot dubbing using a specific optimization artifact:

```bash
./dub.sh input.mp4 en pt --best path/to/best.json
```

See [docs/optimize.md](docs/optimize.md) for commands, flags, the composite
metric, resuming, and storage details.

## Professional Dubbing Tips

### Named Entity Preservation
To prevent characters or brands from being translated (e.g., "Ei Nerd" becoming "Hey Nerd"), use a glossary:
1. Generate a template: `./dub.sh glossary template --output entities.json`
2. Add your terms to the JSON file.
3. Run with: `./dub.sh input.mp4 pt en --glossary entities.json`

## Requirements

- Python 3.12
- FFmpeg on PATH
- An NVIDIA GPU with at least ~3 GB of free VRAM (8 GB recommended)
- Hugging Face token (`HF_TOKEN`) for high-quality pyannote diarization
  (optional — a VAD-based fallback is used otherwise)
