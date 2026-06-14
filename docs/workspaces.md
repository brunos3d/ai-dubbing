# Workspaces

A **workspace** is a persistent, inspectable directory that captures every
intermediate artefact produced while dubbing a single piece of media. Where
the one-shot `dub.sh run` discards most of the pipeline's work inside a
temp-dir cache, a workspace keeps it on disk under
`~/.local/share/ai-dubbing/workspaces/`, so you can read the transcript,
swap a speaker sample, fix a translation, and re-run only the stages that
depend on what you changed.

Full design: `docs/superpowers/specs/2026-06-13-workspace-architecture-design.md`
(§1 problem, §6 file layout, §9 manifest, §10 invalidation algorithm).

## 1. The two-phase workflow

| Phase | Command | Stages run |
|---|---|---|
| `prepare` | `dub.sh prepare <input> <src> <tgt>` | extract → separate → diarize → samples → transcribe → translate |
| `generate` | `dub.sh generate <workspace-id>` | generate → align → reconstruct → mix → video |

`prepare` stops once you have a translated transcript and a speaker profile
for each speaker — everything you might want to edit. `generate` consumes
that work and produces the final dubbed audio/video. You can edit files in
between; `generate` re-runs only the stages affected by your change. The
legacy one-shot `./dub.sh input.mp4 pt en` is unchanged: it runs
`prepare` + `generate` internally and prints the workspace ID at the end.

## 2. Workspace location

```
~/.local/share/ai-dubbing/workspaces/
└── <slug>-<YYYYMMDD>-<hash8>/
```

* **`<slug>`** — input filename stem, lowercased, non-alphanumerics replaced
  with `-`. Override with `--name <slug>`.
* **`<YYYYMMDD>`** — local date of `prepare`.
* **`<hash8>`** — first 8 hex chars of
  `sha256(media_sha256 || src_lang || tgt_lang || pipeline_config_hash)`.

The root holds `manifest.json` and `metadata.json` at the top level, plus
one subdir per stage family: `source/`, `media/`, `diarization/`,
`transcription/`, `translation/`, `speakers/`, `output/`, `logs/`. See §6
of the design spec for the full tree.

## 3. `dub.sh prepare`

```bash
dub.sh prepare <input> <src> <tgt> [--name <slug>] [--glossary <file>]
```

Runs the analysis half of the pipeline and creates (or reuses) a workspace.
Use it when you want to inspect what the pipeline detected, plan to edit
translations/glossaries/transcripts/speaker samples, or batch-prepare
several pieces of media and produce audio later. Stops after `translate`
and prints the workspace ID and on-disk path.

| Flag | Effect |
|---|---|
| `--name <slug>` | override the auto-derived workspace slug |
| `--glossary <file>` | seed `translation/glossary.json` from a template |
| `--no-pyannote` | use the VAD+MFCC fallback for diarization |
| `--min-speakers N` / `--max-speakers N` | constrain the speaker count |
| `--whisper-model <size>` | override the transcription model (default: `large-v3`) |

## 4. `dub.sh generate`

```bash
dub.sh generate <workspace-id> [--from-stage NAME] [--to-stage NAME] [--force]
```

Runs the synthesis half of the pipeline against an existing workspace.
Before re-running, `generate` recomputes SHA-256 hashes for every artefact
and only re-executes the stages that are stale. Omit `<workspace-id>` to
operate on the most recently modified workspace.

| Flag | Effect |
|---|---|
| `--from-stage NAME` | force a run starting at this stage, regardless of staleness |
| `--to-stage NAME` | cap the run at this stage (inclusive); useful for debugging |
| `--force` | re-run every stage; ignore the DAG |

```bash
dub.sh generate                                          # resume the latest workspace
dub.sh generate myvideo-20260613-b7c1f4aa
dub.sh generate myvideo-20260613-b7c1f4aa --to-stage mix  # stop before video remux
dub.sh generate myvideo-20260613-b7c1f4aa --from-stage generate
dub.sh generate myvideo-20260613-b7c1f4aa --force
```

## 5. `dub.sh workspace` subcommands

```bash
dub.sh workspace list                            # table of all workspaces
dub.sh workspace inspect <id>                    # manifest summary + stage statuses
dub.sh workspace show <id> [path]                # print a subdir path (or all)
dub.sh workspace show <id> translation           # -> …/workspaces/<id>/translation
dub.sh workspace validate <id>                   # run every per-artifact validator
dub.sh workspace open <id>                       # print the workspace root
dub.sh workspace clean <id> [--keep-outputs] [--yes]
```

`list` reads each workspace's `metadata.json`; `inspect` reads
`manifest.json` and prints the stage table. `validate` calls one validator
per known artefact type and exits non-zero on any error. `clean` asks for
confirmation (use `--yes` to skip) and either deletes the whole workspace
or, with `--keep-outputs`, only the `output/` subdir.

## 6. User-editable artefacts

Five files are safe to edit by hand. `generate` will detect the change and
re-run only the stages that depend on it.

| Path | What you can change |
|---|---|
| `translation/translated_transcript.json` | edit a segment's `text` (and optionally `source_text`) |
| `translation/glossary.json` | add a new entry: `"Name": { "action": "preserve" }` |
| `transcription/transcript.json` | fix a misheard word in a segment's `text` |
| `speakers/<id>/primary.wav` | swap a speaker's reference audio (mono 16 kHz, 8–12 s preferred, ≤ 15 s) |
| `speakers/<id>/primary.txt` | edit the reference transcript (must match the audio) |

The first three are text edits. Speaker samples are validated pre-flight:
duration under 3 s or above 15 s, or a sample rate other than 16 kHz, is a
warning. A `primary.txt` is required when a `primary.wav` is present.

## 7. Three artefact classes

| Class | Recorded in | Hash-mismatch behaviour |
|---|---|---|
| **editable** | top-level `editable_paths` | mark the *consumer* stage stale; trust the user's edit (no upstream walk) |
| **non-editable** (default) | per-stage `outputs` | mark the *producer* stage stale (and downstream by propagation) |
| **derived** | top-level `derived_paths` | not part of invalidation; regenerated as a side-effect of its owning stage |

The five files in §6 are the only `editable_paths`. `diarization/embeddings.npz`
and `logs/pipeline.log` are derived — recreated when their owning stage runs
but never invalidate anything. Everything else is non-editable: a hash
mismatch triggers a full upstream rebuild.

## 8. Common workflows

```bash
WS=$(dub.sh workspace list | awk 'NR==3{print $1}')    # grab the latest ID
```

**Fix a bad translation**

```bash
$EDITOR ~/.local/share/ai-dubbing/workspaces/$WS/translation/translated_transcript.json
dub.sh generate $WS                                     # only generate..video re-runs
```

**Add a glossary entry after the fact**

```bash
$EDITOR ~/.local/share/ai-dubbing/workspaces/$WS/translation/glossary.json
dub.sh generate $WS                                     # translate, then generate..video re-run
```

**Swap a speaker sample**

```bash
SPEAKER=$(ls ~/.local/share/ai-dubbing/workspaces/$WS/speakers | head -1)
dub.sh workspace validate $WS                          # confirm the new wav passes
cp /path/to/better-sample.wav \
   ~/.local/share/ai-dubbing/workspaces/$WS/speakers/$SPEAKER/primary.wav
dub.sh generate $WS                                     # only generate..video re-runs
```

**Throw away the output and re-generate**

```bash
dub.sh workspace clean <id> --keep-outputs --yes
dub.sh generate <id>
```

## 9. Environment variables

| Variable | Effect |
|---|---|
| `AI_DUBBING_WORKSPACES_ROOT` | absolute override for the workspaces root directory |
| `XDG_DATA_HOME` | if set, the root resolves to `$XDG_DATA_HOME/ai-dubbing/workspaces` |
| `HF_TOKEN` | Hugging Face token for gated pyannote models; its *presence* is recorded as `hf_token_available: bool` in `metadata.json`, but the value itself is **never** part of the workspace hash |

When none of these are set, the default is
`~/.local/share/ai-dubbing/workspaces/` (XDG default).
