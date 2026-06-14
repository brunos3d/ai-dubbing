# Workspace Architecture — Design Spec

**Date:** 2026-06-13
**Status:** Draft
**Author:** opencode

## 1. Problem

The ai-dubbing pipeline is a black box:

```
Input
  ↓
Pipeline
  ↓
Final video/audio
```

The pipeline already generates valuable intermediate artifacts (extracted audio, separated speech, speaker diarization, voice profiles, transcripts, translations) but throws them away inside a temp-dir cache that gets auto-pruned after 7 days. Users cannot:

* inspect what the pipeline detected
* replace a poor speaker sample without re-running diarization
* fix a bad translation without re-transcribing
* add glossary entries after the fact
* reuse a workspace on a different machine
* hand a workspace to a collaborator for review

The result: every "minor" edit forces a full pipeline re-run, and the user has no way to understand *why* a dub sounds the way it does.

## 2. Goals

1. **Two-phase workflow.** `prepare` runs all analysis stages and stops; `generate` consumes the prepared artifacts and produces the final output.
2. **Inspectable artifacts.** Every intermediate file lives in a human-readable, JSON/TXT/WAV-based directory layout that a future UI (Electron, Next.js, REST) can read with no pipeline knowledge.
3. **Edit-then-resume.** Replacing a `primary.wav` or editing a `translated_transcript.json` re-runs *only* the dependent stages — not the entire pipeline.
4. **Stable workspace identity.** Each workspace has a deterministic, shareable ID derived from its content + config, so the same input always lands on the same workspace.
5. **Clean separation of concerns.** Temporary cache (one-shot execution) and persistent workspace (user asset) are distinct systems.
6. **Backwards compatible.** `dub.sh run` keeps working exactly as before. The workspace is created transparently and its ID is printed at the end.

## 3. Non-goals

* A graphical UI. (Designed-for, not built-now.)
* A REST API. (Designed-for, not built-now.)
* Distributed / multi-machine workspace execution.
* Automatic workspace cleanup. (User-managed via `dub.sh workspace clean`.)
* Replacing the existing temp-dir cache. (The two systems coexist.)
* Speaker *identification* (matching against a known voiceprint). The system supports *editing* a speaker profile, not *verifying* it.

## 4. Architecture overview

```
                 ┌──────────────────────────────────────────┐
                 │ WorkspacePipeline (new, src/workspace/)  │
                 │   - workspace lifecycle                  │
                 │   - manifest + DAG                       │
                 │   - invalidation algorithm               │
                 │   - atomic stage writes                  │
                 │   - config hashing                       │
                 │   - editable/derived tracking            │
                 └────────────────────┬─────────────────────┘
                                      │ calls
                 ┌────────────────────▼─────────────────────┐
                 │ Pipeline (existing, src/pipeline.py)     │
                 │   - stage ordering                       │
                 │   - execution flow                       │
                 │   - checkpoint (internal only)           │
                 └────────────────────┬─────────────────────┘
                                      │ invokes
                 ┌────────────────────▼─────────────────────┐
                 │ Stage implementations (src/stages/*.py) │
                 │   - business logic only                  │
                 │   - declare inputs/outputs/metadata      │
                 └──────────────────────────────────────────┘
```

Three layers, three responsibilities:

* **Stage** — *what* to do (extract audio, diarize, translate, …).
* **Pipeline** — *how* to execute stages in order.
* **WorkspacePipeline** — *when* and *why* to execute each stage (freshness, dependency, invalidation).

## 5. Command structure

```
dub.sh prepare <input> <src> <tgt> [--name <slug>] [--glossary <file>]
    Runs extract → separate → diarize → samples → transcribe → translate.
    Stops. Creates/loads a workspace. Prints workspace ID + path.

dub.sh generate <workspace-id> [--from-stage <name>] [--to-stage <name>] [--force]
    Loads the workspace. Runs invalidation analysis.
    Re-runs only the affected stages from `generate` onwards (or --from-stage).
    Produces output/final_audio.wav and/or output/final_video.mp4.

dub.sh workspace list
    Lists all workspaces with id, source, languages, created, status.

dub.sh workspace inspect <workspace-id>
    Prints manifest summary: stages, modified artifacts, will-reuse vs will-rerun.

dub.sh workspace show <workspace-id> [path]
    Prints paths to key artifacts (for editing or for a UI to consume).

dub.sh workspace open <workspace-id>
    Prints the workspace path. (Future: xdg-open / Electron handler.)

dub.sh workspace validate <workspace-id>
    Runs all per-artifact validators. Prints errors and warnings.

dub.sh workspace clean <workspace-id> [--keep-outputs]
    Removes a workspace (with confirmation).

dub.sh run <input> <src> <tgt> [options]    # existing, unchanged UX
    Internally: prepare + generate. Prints the workspace ID at the end.
```

**Stage boundary:** `prepare` stops after `translate`. `generate` runs `generate → align → reconstruct → mix → video`.

## 6. Workspace file structure

```
~/.local/share/ai-dubbing/workspaces/
└── peter-ei-nerd-20260613-b7c1f4aa/        ← human-readable slug + date + short hash
    │
    ├── manifest.json              ← DAG: stages, inputs, outputs, hashes
    ├── metadata.json              ← workspace identity, source, languages, versions
    │
    ├── source/
    │   └── peter-ei-nerd.mp4      ← symlink to original input (reproducibility)
    │
    ├── media/                     ← from extract + separate stages
    │   ├── original_audio.wav
    │   ├── speech.wav
    │   └── background.wav
    │
    ├── diarization/               ← from diarize stage
    │   ├── segments.json
    │   ├── embeddings.npz         ← (n_chunks, 256) float32 + labels + times
    │   ├── embeddings.meta.json
    │   └── metadata.json          ← model, k, min/max speakers, scoring trace
    │
    ├── transcription/             ← from transcribe stage
    │   ├── transcript.json        ★ user-editable
    │   └── word_level.json
    │
    ├── translation/               ← from translate stage
    │   ├── translated_transcript.json   ★ user-editable
    │   └── glossary.json                ★ user-editable
    │
    ├── speakers/                  ← from samples stage
    │   ├── speaker_01/
    │   │   ├── primary.wav        ★ user-editable (8-12s, mono 16kHz)
    │   │   ├── primary.txt        ★ user-editable
    │   │   ├── embedding.npy      ← 256-dim float32 centroid
    │   │   ├── metadata.json
    │   │   └── candidates/
    │   │       ├── candidate_01.wav
    │   │       ├── candidate_01.txt
    │   │       └── candidate_01.score.json
    │   └── speaker_02/
    │       └── ...
    │
    ├── output/                    ← from mix + video stages
    │   ├── final_audio.wav
    │   └── final_video.mp4
    │
    └── logs/
        └── pipeline.log
```

**`★` = user-editable.** Everything else is regenerated when its inputs change.

## 7. Workspace ID generation

```
<source-slug>-<YYYYMMDD>-<hash8>/
```

- **`<source-slug>`** — input filename stem, lowercased, non-alphanumerics replaced with `-`, consecutive `-` collapsed, trimmed. Override with `--name <custom>`.
- **`<YYYYMMDD>`** — local date of `prepare`.
- **`<hash8>`** — first 8 hex chars of `content_hash`.

**`content_hash` = `sha256(media_sha256 || src_lang || tgt_lang || pipeline_config_hash)`.**

`pipeline_config_hash` is `sha256` of a deterministic JSON dump of all stage configurations (model names, speaker limits, translation provider, voice-design flags, …). The full config is *recorded* in `metadata.json` and `manifest.json`; the hash is what determines identity.

The `hf_token` is **never** part of `content_hash` — rotating a token or switching HuggingFace accounts must not invalidate workspaces. The *presence* of a token is recorded in `metadata.json` as `hf_token_available: bool`.

## 8. `metadata.json` schema

```json
{
  "workspace_schema_version": 1,
  "workspace_id": "peter-ei-nerd-20260613-b7c1f4aa",
  "content_hash": "b7c1f4aa...",
  "created_at": "2026-06-13T12:34:56Z",
  "updated_at": "2026-06-13T12:40:12Z",
  "workspace_created_with": {
    "pipeline_version": "1.2.3",
    "git_commit": "328942c",
    "python_version": "3.12.4"
  },
  "source": {
    "media_path": "/abs/path/peter-ei-nerd.mp4",
    "media_sha256": "abc...",
    "source_language": "pt",
    "target_language": "en"
  },
  "config": {
    "whisper_model": "large-v3",
    "hf_token_available": true,
    "min_speakers": null,
    "max_speakers": 8,
    "no_pyannote": false,
    "target_lufs": -16.0,
    "glossary_path": "/abs/path/entities.json"
  },
  "pipeline_config_hash": "def..."
}
```

`workspace_schema_version` exists so future versions of the tool can detect and migrate old workspaces.

## 9. `manifest.json` schema and dependency DAG

`manifest.json` is the authoritative state file. It replaces `checkpoint.json`. The DAG is implicit: each stage's `inputs` reference paths produced by other stages.

```json
{
  "schema_version": 1,
  "pipeline_version": "1.2.3",
  "git_commit": "328942c",
  "editable_paths": [
    "transcription/transcript.json",
    "translation/translated_transcript.json",
    "translation/glossary.json",
    "speakers/*/primary.wav",
    "speakers/*/primary.txt"
  ],
  "derived_paths": [
    "diarization/embeddings.npz",
    "logs/pipeline.log"
  ],
  "stages": {
    "extract": {
      "status": "done",
      "config": {},
      "started_at": "2026-06-13T12:34:56Z",
      "finished_at": "2026-06-13T12:35:08Z",
      "duration_s": 12.3,
      "inputs": [],
      "outputs": [
        {"path": "media/original_audio.wav", "sha256": "abc...", "size_bytes": 12345}
      ]
    },
    "separate": {
      "status": "done",
      "config": {},
      "inputs": [{"path": "media/original_audio.wav", "sha256": "abc..."}],
      "outputs": [
        {"path": "media/speech.wav", "sha256": "def..."},
        {"path": "media/background.wav", "sha256": "ghi..."}
      ]
    },
    "diarize": {
      "status": "done",
      "config": {"pyannote": true, "min_speakers": null, "max_speakers": 8},
      "inputs": [{"path": "media/speech.wav", "sha256": "def..."}],
      "outputs": [
        {"path": "diarization/segments.json", "sha256": "..."},
        {"path": "diarization/embeddings.npz", "sha256": "..."},
        {"path": "diarization/embeddings.meta.json", "sha256": "..."},
        {"path": "diarization/metadata.json", "sha256": "..."}
      ]
    },
    "samples": {
      "status": "done",
      "config": {"target_seconds": 10.0, "max_seconds": 15.0},
      "inputs": [
        {"path": "media/speech.wav", "sha256": "def..."},
        {"path": "diarization/segments.json", "sha256": "..."}
      ],
      "outputs": [
        {"path": "speakers/speaker_01/primary.wav", "sha256": "...", "editable": true},
        {"path": "speakers/speaker_01/primary.txt", "sha256": "...", "editable": true},
        ...
      ]
    },
    "transcribe": { "...": "..." },
    "translate": { "...": "..." },
    "generate": { "...": "..." },
    "align": { "...": "..." },
    "reconstruct": { "...": "..." },
    "mix": { "...": "..." },
    "video": { "...": "..." }
  }
}
```

**Three artifact classifications:**

| Class | Manifest field | Hash-mismatch behavior |
|---|---|---|
| **editable** | `editable: true` on output, listed in top-level `editable_paths` | Mark the *consumer* stage as stale. Do **not** walk upstream — the user's edit is trusted. |
| **non-editable** (default) | recorded as a plain output | Mark the *producing* stage as stale (transitively), then the consumer. |
| **derived** | listed in top-level `derived_paths` | Not part of invalidation. Regenerated as a side-effect of its owning stage. |

**Per-stage `config`** is a deterministic dict; a config change invalidates the stage even if no file hash changed (covers: model name change, speaker-limit change, translation-provider change, voice-design flag change).

## 10. Invalidation algorithm

Triggered by `dub.sh generate` (and by the second half of `dub.sh run`).

```
def compute_stale_set(manifest, workspace_root, cli_overrides):
    stale = set()
    config_stale = set()

    # 1. Recompute hashes for all referenced paths
    actual_hashes = {}
    for stage_name, stage in manifest.stages.items():
        for io in stage.inputs + stage.outputs:
            p = workspace_root / io.path
            if p.exists():
                actual_hashes[io.path] = sha256(p)

    # 2. Detect hash-mismatches and config-mismatches
    for stage_name, stage in manifest.stages.items():
        for io in stage.inputs:
            recorded = io.sha256
            current = actual_hashes.get(io.path)
            if current is None or current != recorded:
                if is_editable(io.path, manifest):
                    stale.add(stage_name)            # trust the user's edit
                else:
                    # walk upstream to the producing stage
                    producer = find_producer(io.path, manifest)
                    if producer:
                        stale.add(producer)
                    stale.add(stage_name)
        if stage_has_config_change(stage_name, manifest):
            stale.add(stage_name)

    # 3. Propagate downstream: any stage whose input is produced by a
    #    stale stage is itself stale.
    changed = True
    while changed:
        changed = False
        for stage_name, stage in manifest.stages.items():
            if stage_name in stale:
                continue
            for io in stage.inputs:
                if is_derived(io.path, manifest):
                    continue
                producer = find_producer(io.path, manifest)
                if producer in stale and stage_name not in stale:
                    stale.add(stage_name)
                    changed = True

    # 4. CLI overrides
    if cli_overrides.force:
        stale = set(manifest.stages.keys())
    if cli_overrides.from_stage:
        from_idx = STAGE_ORDER.index(cli_overrides.from_stage)
        for s in STAGE_ORDER[from_idx:]:
            stale.add(s)
    if cli_overrides.to_stage:
        to_idx = STAGE_ORDER.index(cli_overrides.to_stage)
        stale = {s for s in stale if STAGE_ORDER.index(s) <= to_idx}

    # 5. Sort topologically
    return [s for s in STAGE_ORDER if s in stale]
```

**Verified scenarios:**

| User edit | Expected stale set |
|---|---|
| `translation/translated_transcript.json` (editable) | generate, align, reconstruct, mix, video |
| `speakers/speaker_03/primary.wav` (editable) | generate, align, reconstruct, mix, video |
| `translation/glossary.json` (editable) | translate, generate, align, reconstruct, mix, video |
| `transcription/transcript.json` (editable) | translate, generate, align, reconstruct, mix, video |
| `diarization/segments.json` (non-editable) | diarize, samples, transcribe, translate, generate, align, reconstruct, mix, video |
| `media/speech.wav` (non-editable) | diarize, samples, transcribe, translate, generate, align, reconstruct, mix, video |
| `--from-stage generate` | generate, align, reconstruct, mix, video (regardless of hashes) |
| `--force` | all stages |

## 11. Atomic stage writes

1. Orchestrator creates `<workspace>/.tmp/<stage-name>-<uuid>/`
2. Stage writes all its outputs into the temp dir (paths preserved relative to workspace root)
3. Orchestrator computes SHA-256 of each output
4. Orchestrator atomically renames temp dir contents into the workspace, one file at a time
5. Orchestrator updates `manifest.json` with new hashes + status
6. Temp dir is removed

A failed stage leaves the workspace consistent with the last successful stage. The next invocation re-runs from the failed stage.

## 12. Artifact formats

### Transcript / translated-transcript

```json
{
  "$schema_version": 1,
  "language": "pt",
  "segments": [
    {
      "id": "seg_0001",
      "start": 0.52,
      "end": 3.18,
      "speaker": "speaker_01",
      "text": "Olá mundo",
      "source_text": "Olá mundo",
      "translation_confidence": 0.97,
      "words": [
        {"start": 0.52, "end": 0.91, "text": "Olá", "confidence": 0.99}
      ]
    }
  ]
}
```

`source_text` makes each translated segment self-contained (UI doesn't need to join with `transcript.json`). `translation_confidence` is reserved for future backends that supply it.

### Glossary

```json
{
  "$schema_version": 1,
  "entries": {
    "Ei Nerd":    {"action": "preserve"},
    "Spider-Man": {"action": "preserve"},
    "API":        {"action": "preserve"}
  }
}
```

### Diarization segments

```json
{
  "$schema_version": 1,
  "speakers": ["speaker_01", "speaker_02"],
  "segments": [
    {"start": 0.52, "end": 3.18, "speaker": "speaker_01", "low_confidence": false}
  ]
}
```

### Diarization embeddings

`diarization/embeddings.npz` — three arrays in one compressed file:

* `vectors` — `np.ndarray` shape `(n_chunks, 256)`, dtype `float32`
* `labels` — `np.ndarray` shape `(n_chunks,)`, dtype `int32`
* `times` — `np.ndarray` shape `(n_chunks, 2)`, dtype `float32`

`diarization/embeddings.meta.json` — model name, dim, extracted_at.

### Speaker embedding

`speakers/<id>/embedding.npy` — single 256-dim `float32` array.

### Speaker metadata

```json
{
  "$schema_version": 1,
  "speaker_id": "speaker_01",
  "voice_profile_hash": "sha256:abc...",
  "score": 87.4,
  "snr_db": 28.1,
  "reference_duration_s": 10.4,
  "segments_used": 3,
  "transcript_words": 27,
  "model": "wespeaker-resnet34",
  "extracted_at": "2026-06-13T12:34:56Z"
}
```

`voice_profile_hash` = SHA-256 of the *content* of `primary.wav` + `primary.txt` + `embedding.npy` concatenated. Stable identity for profile reuse and comparison.

### Speaker primary.wav requirements

* Mono, 16 kHz PCM
* 8–12 s preferred
* ≤ 15 s maximum
* Validated pre-flight: fail-fast on duration < 3 s, RMS below threshold, SNR below threshold

### Speaker candidate score

`speakers/<id>/candidates/candidate_<n>.score.json`:

```json
{
  "$schema_version": 1,
  "candidate_id": "candidate_01",
  "score": 72.1,
  "snr_db": 24.3,
  "duration_s": 9.2,
  "segments_used": 2,
  "model": "wespeaker-resnet34"
}
```

## 13. Stage refactor — minimal changes to existing code

The existing `src/pipeline.py` is kept as-is. A new `src/workspace/` package wraps it.

Each stage gains declarative metadata:

```python
class SampleStage:
    name = "samples"
    inputs = ["media/speech.wav", "diarization/segments.json"]
    outputs = [
        "speakers/*/primary.wav",
        "speakers/*/primary.txt",
        "speakers/*/embedding.npy",
        "speakers/*/metadata.json",
        "speakers/*/candidates/*",
    ]
    editable_outputs = [
        "speakers/*/primary.wav",
        "speakers/*/primary.txt",
    ]
    config_fields = ["target_seconds", "max_seconds"]
    # run() unchanged in body; receives WorkspaceContext, returns dict
```

The stage's `__init__` no longer takes `workdir`; paths come from `WorkspaceContext`. The Pipeline still orchestrates execution order, but `WorkspacePipeline` wraps it to handle the workspace, manifest, invalidation, and atomic writes.

## 14. Validation

`src/workspace/validate.py` exports one function per artifact type:

* `validate_transcript(data) -> list[Issue]`
* `validate_translation(data) -> list[Issue]`
* `validate_glossary(data) -> list[Issue]`
* `validate_diarization_segments(data) -> list[Issue]`
* `validate_speaker_metadata(data) -> list[Issue]`
* `validate_speaker_sample(wav_path, txt_path) -> list[Issue]`
* `validate_manifest(data) -> list[Issue]`
* `validate_metadata(data) -> list[Issue]`

Each returns a list of `Issue(severity: "error"|"warning", path: str, message: str)`. Errors block generation; warnings do not.

Called automatically by `generate` before the invalidation pass. Exposed as `dub.sh workspace validate <id>`.

## 15. Error handling

| Failure | Behavior |
|---|---|
| Stage throws | Mark stage `failed` in manifest, leave workspace consistent with last successful stage, exit 1 |
| Validation error | Exit 2 with file path + reason |
| Validation warning | Print but continue |
| Hash mismatch mid-run | Another process modified a workspace file; abort, suggest re-run |
| Workspace ID not found | Print `No workspace with ID 'foo'. Run 'dub.sh workspace list' to see available workspaces.` |
| Insufficient disk | Pre-flight check: warn if < 1 GB free; abort if < 200 MB. Estimate = source media size + 0.3× source for separated stems + 50 MB / min for embeddings + estimated final output (3 MB/min for wav, 1 MB/min for mp4). Print estimate before processing long media. |
| GPU OOM | Catch `torch.cuda.OutOfMemoryError`, free VRAM, mark stage `failed` in manifest, **preserve workspace state**, print suggestion. User can re-run `dub.sh generate <id>` after adjusting settings. |

## 16. Migration

**No migration.** The existing `/tmp/ai-dubbing/<key>/` cache stays exactly as it is. The new `workspace` subcommand is a parallel system. The two systems coexist:

* `dub.sh cache ...` — temp cache (auto-pruned, project-hash-keyed, implementation detail)
* `dub.sh workspace ...` — persistent workspaces (user-named, content-hash-keyed, user asset)

`dub.sh run` is updated internally to invoke `WorkspacePipeline.prepare()` + `WorkspacePipeline.generate()` and to print the workspace ID at the end. One-shot users get a workspace transparently.

## 17. Testing strategy

**Release blocker:** `tests/test_workspace_dag.py` covers every edit scenario from §10. One test case per row in the table.

**Unit:**
* `test_workspace_paths.py` — slug generation, root path, ID parsing
* `test_workspace_manifest.py` — load/save round-trip, schema_version handling
* `test_workspace_dag.py` — invalidation algorithm against §10 table
* `test_workspace_validate.py` — every validator function (valid + invalid fixtures)
* `test_workspace_atomic.py` — temp-dir staging, rename, cleanup on failure
* `test_workspace_id.py` — `content_hash` determinism, slug edge cases

**Integration:**
* `test_workspace_e2e.py` — full prepare on a 30-second test fixture; assert all expected files + valid schemas
* `test_workspace_edit_scenarios.py` — synthetic workspace + manifest; apply each edit from §10; assert correct stale set; assert only the right stages re-execute

**Backwards compat:**
* All existing tests pass
* `dub.sh run` produces the same outputs as before
* `dub.sh cache list` still works

**Test fixtures:**
* `tests/fixtures/short_sample.wav` — 30 s synthetic test audio (deterministic sine + silence)

## 18. Files

**NEW**
* `src/workspace/__init__.py`
* `src/workspace/paths.py` — `~/.local/share/ai-dubbing/workspaces` root, slug generation, ID parsing
* `src/workspace/manifest.py` — Manifest dataclass, load/save, atomic write
* `src/workspace/dag.py` — invalidation algorithm
* `src/workspace/atomic.py` — temp-dir staging + atomic rename
* `src/workspace/validate.py` — per-artifact validators
* `src/workspace/content_hash.py` — `content_hash` and `pipeline_config_hash`
* `src/workspace/pipeline.py` — `WorkspacePipeline` (wraps `Pipeline`)
* `src/workspace/cli.py` — workspace subcommand handlers
* `tests/test_workspace_paths.py`
* `tests/test_workspace_manifest.py`
* `tests/test_workspace_dag.py`
* `tests/test_workspace_validate.py`
* `tests/test_workspace_atomic.py`
* `tests/test_workspace_id.py`
* `tests/test_workspace_e2e.py`
* `tests/test_workspace_edit_scenarios.py`
* `tests/fixtures/short_sample.wav`
* `docs/workspaces.md` — user-facing workspace guide

**MODIFIED**
* `src/cli.py` — add `prepare`, `generate`, `workspace` subcommands; thread workspace plumbing into `run`
* `src/pipeline.py` — keep as-is; add `pipeline_config_dict()` for hashing
* `src/stages/*.py` — add `inputs`/`outputs`/`editable_outputs`/`derived_outputs`/`config_fields` class attributes; switch to `WorkspaceContext`
* `dub.sh` — add top-level `prepare` and `generate`; add `workspace` subcommand handler; auto-print workspace ID at end of `run`
* `README.md` — document new commands and the workspace concept

**UNCHANGED**
* `src/utils/cache.py` — temp cache stays separate
* `src/utils/checkpoint.py` — still used internally by `Pipeline`

## 19. Out of scope (deferred)

* Speaker profile library / cross-workspace reuse (only `voice_profile_hash` is reserved for it)
* REST API on top of `WorkspacePipeline`
* Electron / Next.js UI
* Workspace export/import as a tarball (for sharing with collaborators)
* Distributed generation (multi-GPU)
* Auto-cleanup of old workspaces
