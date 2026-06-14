# Workspace Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the prepare/generate workspace architecture end-to-end per `docs/superpowers/specs/2026-06-13-workspace-architecture-design.md`, with TDD.

**Architecture:** A new `src/workspace/` package wraps the existing `Pipeline` to add a two-phase workflow (`prepare` stops after `translate`; `generate` runs `generate..video`). Workspaces live at `~/.local/share/ai-dubbing/workspaces/<slug>-<YYYYMMDD>-<hash8>/`. A `manifest.json` tracks per-stage inputs/outputs with SHA-256 hashes; an invalidation DAG (spec §10) re-runs only stale stages after a user edits an editable file. Stage writes are atomic (temp-dir staging + rename). `dub.sh run` keeps working with the same UX (one extra workspace-ID line at the end).

**Tech Stack:** Python 3.12, pytest, numpy + soundfile (for synthetic test fixture), existing pipeline + stages.

---

## Conventions

* One commit per task. Conventional commits (`feat:`, `test:`, `refactor:`, `chore:`, `docs:`, `fix:`).
* Tests live in `tests/`. Each test module prepends `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` so `from src...` works without install.
* All hashes are hex SHA-256 strings.
* `WorkspaceContext` (Task 10) is a dataclass that is *also* a dict (via `__getitem__`/`__setitem__`/`.get`) so existing `context["audio_path"]` lookups keep working.
* Each stage keeps its existing `__init__(self, workdir, ...)` shape and gains class attributes `name`, `inputs`, `outputs`, `editable_outputs`, `derived_outputs`, `config_fields`. The WorkspacePipeline builds stages with `workdir = workspace_root` and an explicit `subdir=<workspace-relative-dir>`, so stages write into the nested workspace layout (`workspace_root / "media" / "original_audio.wav"`). The legacy `Pipeline._build_stage()` continues to use a flat `workdir` so existing tests do not break.
* The path-from-stage mapping (`extract` -> `media`, `diarize` -> `diarization`, etc.) lives in `src/workspace/pipeline.py:STAGE_SUBDIR` and is tested in `tests/test_workspace_pipeline_helpers.py`.
* `dub.sh run` is **not** refactored to use WorkspacePipeline (would require a complete migration). It continues to use the legacy `Pipeline`; the workspace-ID line is appended by `dub.sh` itself (Task 13) by reading `~/.local/share/ai-dubbing/workspaces/` directly.

---

## File map

**New**

* `src/workspace/__init__.py`
* `src/workspace/paths.py` — slug, root, ID parsing
* `src/workspace/content_hash.py` — content_hash + pipeline_config_hash
* `src/workspace/manifest.py` — Manifest dataclass, atomic load/save
* `src/workspace/atomic.py` — temp-dir staging + atomic rename
* `src/workspace/validate.py` — per-artifact validators
* `src/workspace/dag.py` — invalidation algorithm (spec §10)
* `src/workspace/pipeline.py` — WorkspacePipeline (wraps Pipeline)
* `src/workspace/cli.py` — workspace subcommand handlers
* `tests/test_workspace_paths.py`
* `tests/test_workspace_manifest.py`
* `tests/test_workspace_dag.py` (RELEASE BLOCKER)
* `tests/test_workspace_validate.py`
* `tests/test_workspace_atomic.py`
* `tests/test_workspace_id.py`
* `tests/test_workspace_e2e.py`
* `tests/test_workspace_edit_scenarios.py`
* `tests/test_workspace_stage_metadata.py`
* `tests/test_workspace_pipeline_helpers.py`
* `tests/test_workspace_cli.py`
* `tests/test_cli_subcommands.py`
* `tests/test_pipeline_config_dict.py`
* `tests/test_workspace_short_sample_fixture.py`
* `tests/fixtures/generate_short_sample.py` (helper)
* `tests/fixtures/short_sample.wav` (binary)
* `docs/workspaces.md` — user-facing guide

**Modified**

* `src/cli.py` — add `prepare`, `generate`, `workspace` subcommands
* `src/pipeline.py` — add `pipeline_config_dict()`
* `src/stages/{extract,separate,diarize,samples,transcribe,translate,generate,align,reconstruct,mix,video}.py` — class attrs + `subdir` arg
* `dub.sh` — new top-level commands + workspace ID print
* `README.md` — new commands + workspace concept

---

## Task index

1. Test fixture (30 s synthetic WAV) + lock test
2. `src/workspace/paths.py` + tests
3. `src/workspace/content_hash.py` + tests (`test_workspace_id.py`)
4. `src/workspace/manifest.py` + tests
5. `src/workspace/atomic.py` + tests
6. `src/workspace/validate.py` + tests
7. `src/workspace/dag.py` (RELEASE BLOCKER) + 8 §10 scenario tests
8. Class attributes + `subdir` arg on all 11 stages (`test_workspace_stage_metadata.py`)
9. `pipeline_config_dict()` on `src/pipeline.py`
10. `src/workspace/pipeline.py` (WorkspacePipeline) + helpers test
11. `src/workspace/cli.py` (workspace subcommand handlers)
12. Extend `src/cli.py` with `prepare`, `generate`, `workspace`
13. Update `dub.sh`
14. `tests/test_workspace_e2e.py` + `tests/test_workspace_edit_scenarios.py`
15. `docs/workspaces.md` + `README.md`
16. End-to-end smoke test

The plan deliberately leaves non-trivial code as `**Key points**` summaries plus tests that exercise the *contract* — the full implementation is then written by the executor. See "Style" below.

## Style

For each task below:

1. **TDD step** — show the full test code that the executor must write. Tests are short and complete.
2. **Implementation step** — show key code points and the file the implementation lives in. The executor writes the rest of the implementation to make the tests pass. The full code is required; "key points" are just to keep this plan compact.
3. **Run step** — exact pytest command.
4. **Commit step** — exact commit message.

The full code for each task fits on one or two screens. The plan is intentionally compact; the spec is the source of truth for design decisions.

---

# Task 1: Test fixture — 30 s synthetic WAV

**Files:**

* Create: `tests/fixtures/generate_short_sample.py` (helper script)
* Create: `tests/fixtures/short_sample.wav` (binary output)
* Create: `tests/test_workspace_short_sample_fixture.py`

- [ ] **Step 1: Write the generator script** — `tests/fixtures/generate_short_sample.py`. Signal: 5 s 220 Hz, 5 s silence, 5 s 440 Hz, 5 s silence, 5 s 660 Hz, 5 s silence. Mono 16 kHz PCM_16. Use `numpy` + `soundfile.write`. Output: `tests/fixtures/short_sample.wav`.

- [ ] **Step 2: Run the generator** — `cd /home/bruno/github/tests/ai-dubbing && .venv/bin/python tests/fixtures/generate_short_sample.py`. Expect: `wrote tests/fixtures/short_sample.wav (30s @ 16000Hz mono)`.

- [ ] **Step 3: Write the lock test** — `tests/test_workspace_short_sample_fixture.py` with three tests:
  * `test_fixture_exists` — `assert FIXTURE.exists()` where `FIXTURE = Path(__file__).resolve().parent / "fixtures" / "short_sample.wav"`.
  * `test_fixture_is_mono_16k_pcm16` — open with `soundfile.read(always_2d=True, dtype="float32")`; assert `sr == 16000`, `data.shape[0] == 1`, `data.shape[1] == 30 * 16000`.
  * `test_fixture_sha256_is_64_hex` — `hashlib.sha256(FIXTURE.read_bytes()).hexdigest()`; assert length is 64 and is hex.

- [ ] **Step 4: Run the test green** — `.venv/bin/python -m pytest tests/test_workspace_short_sample_fixture.py -v`. Expect 3 passed.

- [ ] **Step 5: Commit** — `git add tests/fixtures/ tests/test_workspace_short_sample_fixture.py && git commit -m "test: add 30s synthetic WAV fixture + lock test"`.

---

# Task 2: `src/workspace/paths.py`

**Files:**

* Create: `src/workspace/__init__.py`
* Create: `src/workspace/paths.py`
* Create: `tests/test_workspace_paths.py`

- [ ] **Step 1: Write `src/workspace/__init__.py`** — re-export `workspaces_root`, `slugify`, `workspace_id`, `parse_workspace_id`, `WorkspacePathError`.

- [ ] **Step 2: Write the failing test** — `tests/test_workspace_paths.py` with these tests:
  * `test_workspaces_root_default_is_local_share(monkeypatch, tmp_path)` — set `HOME=tmp_path`, unset `XDG_DATA_HOME`; assert root is `tmp_path/.local/share/ai-dubbing/workspaces`.
  * `test_workspaces_root_respects_xdg_data_home(monkeypatch, tmp_path)` — set `XDG_DATA_HOME=tmp_path/xdg`; assert root is `tmp_path/xdg/ai-dubbing/workspaces`.
  * `test_workspaces_root_override(monkeypatch, tmp_path)` — set `AI_DUBBING_WORKSPACES_ROOT=tmp_path/myroot`; assert root is `tmp_path/myroot`.
  * `test_slugify_lowercases_and_replaces_non_alnum` — assert `slugify("Peter Ei-Nerd 2026!!!") == "peter-ei-nerd-2026"`, `slugify("Hello World") == "hello-world"`, `slugify("___") == ""`.
  * `test_workspace_id_format` — call `workspace_id(media_sha256="a"*64, src_lang="pt", tgt_lang="en", pipeline_config_hash="b"*64, source_slug="my-clip", date_yyyymmdd="20260613")`; assert `parts[-2] == "20260613"`, `len(parts[-1]) == 8`, all hex, `"-".join(parts[:-2]) == "my-clip"`.
  * `test_workspace_id_deterministic` — same inputs twice → equal.
  * `test_workspace_id_changes_with_config_hash` — change pipeline_config_hash → different ID.
  * `test_parse_workspace_id_round_trip` — build ID, parse it, assert slug/date/hash8.
  * `test_parse_workspace_id_rejects_garbage` — `parse_workspace_id("not-a-valid-id")` and `parse_workspace_id("slug-20260101-badhex!@")` both raise `WorkspacePathError`.

- [ ] **Step 3: Run — expect failure** — `.venv/bin/python -m pytest tests/test_workspace_paths.py -v`. Expect `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/workspace/paths.py`** — key points:
  * `workspaces_root()`: check `AI_DUBBING_WORKSPACES_ROOT`, then `XDG_DATA_HOME`, then `$HOME/.local/share/ai-dubbing/workspaces`.
  * `slugify(name)`: lowercase, `re.sub(r"[^a-z0-9]+", "-", s)`, `.strip("-")`.
  * `workspace_id(media_sha256, src_lang, tgt_lang, pipeline_config_hash, source_slug, date_yyyymmdd=None)`: compute `content_hash = sha256(f"{media_sha256}{src_lang}{tgt_lang}{pipeline_config_hash}")`; return `f"{slugify(source_slug) or 'workspace'}-{date or today.strftime('%Y%m%d')}-{content_hash[:8]}"`.
  * `parse_workspace_id(s)`: regex `^(?P<slug>.+)-(?P<date>\d{8})-(?P<hash>[0-9a-f]{8})$`; raise `WorkspacePathError` on no match.
  * `WorkspacePathError(ValueError)`.

- [ ] **Step 5: Run — expect green** — `.venv/bin/python -m pytest tests/test_workspace_paths.py -v`. Expect 9 passed.

- [ ] **Step 6: Commit** — `git add src/workspace/__init__.py src/workspace/paths.py tests/test_workspace_paths.py && git commit -m "feat(workspace): paths module (slug, root, workspace_id)"`.

---

# Task 3: `src/workspace/content_hash.py`

**Files:**

* Create: `src/workspace/content_hash.py`
* Create: `tests/test_workspace_id.py`

- [ ] **Step 1: Write the failing test** — `tests/test_workspace_id.py`:
  * `test_media_sha256_matches_stdlib(tmp_path)` — write `b"hello world"`; assert `media_sha256(p) == hashlib.sha256(b"hello world").hexdigest()`.
  * `test_content_hash_excludes_hf_token` — call `content_hash(media_sha256="a"*64, src_lang="en", tgt_lang="es", pipeline_config_hash="b"*64, hf_token="secret")` and again with `hf_token="different"`; assert equal.
  * `test_content_hash_changes_with_inputs` — vary one of `media_sha256`, `src_lang`, `tgt_lang`, `pipeline_config_hash`; assert 5 distinct hashes.
  * `test_pipeline_config_hash_is_deterministic` — same input twice → equal.
  * `test_pipeline_config_hash_key_order_invariant` — `{"a":1,"b":2,"c":3}` and `{"c":3,"b":2,"a":1}` → equal.
  * `test_pipeline_config_hash_changes_with_value` — `{"a":1}` and `{"a":2}` → different.
  * `test_pipeline_config_hash_nested_dicts` — `{"outer":{"a":1,"b":2}}` and `{"outer":{"b":2,"a":1}}` → equal.
  * `test_pipeline_config_hash_ignores_hf_token` — both `{"whisper_model":"large-v3","hf_token":"secret"}` and `{"whisper_model":"large-v3","hf_token":"different"}` → equal.

- [ ] **Step 2: Run — expect failure** — expect `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/workspace/content_hash.py`** — key points:
  * `_HASH_EXCLUDE_KEYS = frozenset({"hf_token", "hf_token_available", "offload_dir"})`.
  * `media_sha256(path)` — stream SHA-256 in 64 KB chunks.
  * `pipeline_config_hash(cfg)` — strip excluded keys, `json.dumps(sort_keys=True, separators=(",",":"), default=str)`, SHA-256.
  * `content_hash(*, media_sha256, src_lang, tgt_lang, pipeline_config_hash, hf_token=None)` — `sha256(f"{media_sha256}{src_lang}{tgt_lang}{pipeline_config_hash}".encode()).hexdigest()`; `hf_token` is accepted and ignored (del'd).

- [ ] **Step 4: Run — expect green** — expect 8 passed.

- [ ] **Step 5: Commit** — `git add src/workspace/content_hash.py tests/test_workspace_id.py && git commit -m "feat(workspace): content_hash + pipeline_config_hash (hf_token excluded)"`.

---

# Task 4: `src/workspace/manifest.py`

**Files:**

* Create: `src/workspace/manifest.py`
* Create: `tests/test_workspace_manifest.py`

- [ ] **Step 1: Write the failing test** — `tests/test_workspace_manifest.py`:
  * `test_minimal_manifest_round_trip(tmp_path)` — create, add a stage with one `ArtifactRef`, add editable + derived paths, save, load, assert round-trip.
  * `test_load_missing_returns_none(tmp_path)` — load non-existent → `None`.
  * `test_load_broken_json_raises(tmp_path)` — write `"not json"` → expect `ManifestError`.
  * `test_save_uses_atomic_replace(tmp_path)` — save then assert no `.tmp` left.
  * `test_find_producer_finds_stage_that_emitted_path(tmp_path)` — stage `extract` outputs `media/x.wav`; `m.find_producer("media/x.wav") == "extract"`.
  * `test_find_producer_returns_none_when_not_found(tmp_path)` — non-existent path → `None`.
  * `test_get_input_returns_record(tmp_path)` — `m.get_input("separate", "media/x.wav")` is `None` because `separate` has no inputs.

- [ ] **Step 2: Run — expect failure** — expect import error.

- [ ] **Step 3: Implement `src/workspace/manifest.py`** — key points:
  * `SCHEMA_VERSION = 1`.
  * `ManifestError(ValueError)`.
  * `@dataclass ArtifactRef(path: str, sha256: str, size_bytes: int = 0)` with `to_dict` / `from_dict`.
  * `@dataclass StageRecord(name, status="pending", config={}, started_at=None, finished_at=None, duration_s=None, inputs=[], outputs=[], error=None)` with `to_dict` / `from_dict`.
  * `@dataclass Manifest(schema_version, workspace_id, pipeline_version, git_commit, editable_paths=[], derived_paths=[], stages={})` with:
    * `create(workspace_id, pipeline_version, git_commit) → Manifest`.
    * `add_stage(name, record)`, `add_editable_path(path)`, `add_derived_path(path)`.
    * `find_producer(path) → Optional[str]` (linear scan over stages' outputs).
    * `get_input(stage_name, path) → Optional[ArtifactRef]`.
    * `to_dict()`, `save(path)` (write to `path.with_suffix(suffix + ".tmp")` then `tmp.replace(path)`), `load(path) → Optional[Manifest]` (return `None` if missing; raise `ManifestError` on broken JSON).

- [ ] **Step 4: Run — expect green** — expect 7 passed.

- [ ] **Step 5: Commit** — `git add src/workspace/manifest.py tests/test_workspace_manifest.py && git commit -m "feat(workspace): manifest dataclass + atomic load/save"`.

---

# Task 5: `src/workspace/atomic.py`

**Files:**

* Create: `src/workspace/atomic.py`
* Create: `tests/test_workspace_atomic.py`

- [ ] **Step 1: Write the failing test** — `tests/test_workspace_atomic.py`:
  * `test_stage_staging_dir_lives_under_dot_tmp(tmp_path)` — `stage_staging_dir(tmp_path, "extract")` → `tmp_path/.tmp/extract-XXXXXXXX`.
  * `test_promote_moves_files_atomically(tmp_path)` — put `a.txt` and `sub/b.txt` in staging; `promote(staging, tmp_path)`; assert files at root, staging dir gone.
  * `test_promote_no_op_when_staging_missing(tmp_path)` — no exception.
  * `test_promote_preserves_existing_files(tmp_path)` — existing file at root is unchanged; new file from staging added.
  * `test_sha256_file(tmp_path)` — `b"hello"` → matches stdlib SHA-256.
  * `test_promote_cleans_staging_on_failure(tmp_path)` — `tmp_path/sub` is a *file* (blocker for mkdir); expect `AtomicWriteError` and staging dir removed.

- [ ] **Step 2: Run — expect failure** — expect import error.

- [ ] **Step 3: Implement `src/workspace/atomic.py`** — key points:
  * `AtomicWriteError(RuntimeError)`.
  * `stage_staging_dir(root, stage_name) → Path` — return `root/.tmp/<stage_name>-<uuid8>`.
  * `sha256_file(path)` — 64 KB streaming SHA-256.
  * `promote(staging, root)`:
    1. If staging does not exist, return.
    2. `try: ... finally: shutil.rmtree(staging, ignore_errors=True)`.
    3. Inside try: iterate `staging.iterdir()`; for each entry, if dir → `shutil.move` its files into `root/<name>/...`; if file → `shutil.move(entry, dest)`; `dest.parent.mkdir(parents=True, exist_ok=True)`.
    4. On exception, raise `AtomicWriteError`.

- [ ] **Step 4: Run — expect green** — expect 6 passed.

- [ ] **Step 5: Commit** — `git add src/workspace/atomic.py tests/test_workspace_atomic.py && git commit -m "feat(workspace): atomic stage writes (temp-dir staging + promote)"`.

---

# Task 6: `src/workspace/validate.py`

**Files:**

* Create: `src/workspace/validate.py`
* Create: `tests/test_workspace_validate.py`

- [ ] **Step 1: Write the failing test** — `tests/test_workspace_validate.py` with 10 tests:
  * `test_validate_transcript_ok` — well-formed dict → `[]`.
  * `test_validate_transcript_missing_keys` — `{"segments": []}` → at least one `Issue(severity="error", ...)`.
  * `test_validate_translation_requires_source_text` — segment without `source_text` → warning "missing source_text".
  * `test_validate_glossary_ok` — `{"$schema_version": 1, "entries": {"Peter": {"action": "preserve"}}}` → `[]`.
  * `test_validate_glossary_rejects_unknown_action` — action `"delete"` → error.
  * `test_validate_diarization_segments_ok` — well-formed dict → `[]`.
  * `test_validate_speaker_metadata_ok` — all required keys present, `reference_duration_s=10.0` → `[]`.
  * `test_validate_speaker_sample_short_audio_is_error(tmp_path)` — write a 0.5 s wav with `soundfile.write(np.zeros(8000), 16000)`; expect an error mentioning "duration".
  * `test_validate_manifest_ok(tmp_path)` — create a real Manifest, save, load JSON, validate → `[]`.
  * `test_validate_metadata_ok` — well-formed metadata dict → `[]`.

- [ ] **Step 2: Run — expect failure** — expect import error.

- [ ] **Step 3: Implement `src/workspace/validate.py`** — key points:
  * `@dataclass(frozen=True) Issue(severity, path, message)`.
  * `validate_transcript(data)`: require keys `("$schema_version", "language", "segments")`; check each segment has `id/start/end/speaker/text` and `text` is a string.
  * `validate_translation(data)`: same as transcript but each segment must have `source_text` (warning).
  * `validate_glossary(data)`: require `("$schema_version", "entries")`; each entry's `action` must be `"preserve"`.
  * `validate_diarization_segments(data)`: require `("$schema_version", "speakers", "segments")`; check each segment has `start/end/speaker`; warn on speakers not in declared list.
  * `validate_speaker_metadata(data)`: require the 10 keys from spec §12; `reference_duration_s < 3.0` → error; `> 15.0` → warning.
  * `validate_speaker_sample(wav, txt)`: read wav with `soundfile.read`; check `sr == 16000` (warn), mono (warn), duration `< 3.0` → error / `> 15.0` → warning; check `txt.exists()` else error.
  * `validate_manifest(data)`: require `("schema_version", "workspace_id", "stages")`; `schema_version != 1` → error.
  * `validate_metadata(data)`: require the 7 top-level keys from spec §8.

- [ ] **Step 4: Run — expect green** — expect 10 passed.

- [ ] **Step 5: Commit** — `git add src/workspace/validate.py tests/test_workspace_validate.py && git commit -m "feat(workspace): per-artifact validators"`.

---

# Task 7: `src/workspace/dag.py` (RELEASE BLOCKER — spec §10)

**Files:**

* Create: `src/workspace/dag.py`
* Create: `tests/test_workspace_dag.py`

- [ ] **Step 1: Write the failing test** — `tests/test_workspace_dag.py`. **One test per row of the spec §10 verified-scenarios table (8 rows), plus 3 defensive tests.** Use the manifest factory `_make_full_manifest(workspace: Path) → Manifest` that wires up all 11 stages with deterministic artifact records. Helper `_write_files(workspace)` materialises the files referenced by the manifest.

Tests (each builds a manifest, mutates one file, calls `compute_stale_set`, asserts the expected ordered list):

1. `test_edit_translation_translated_transcript` — assert `["generate", "align", "reconstruct", "mix", "video"]`.
2. `test_edit_speaker_primary_wav` — write `speakers/speaker_03/primary.wav`; assert `["generate", "align", "reconstruct", "mix", "video"]`.
3. `test_edit_translation_glossary` — write `translation/glossary.json`; assert `["translate", "generate", "align", "reconstruct", "mix", "video"]`.
4. `test_edit_transcription_transcript` — write `transcription/transcript.json`; assert `["translate", "generate", "align", "reconstruct", "mix", "video"]`.
5. `test_edit_diarization_segments_json` — write `diarization/segments.json`; assert `["diarize", "samples", "transcribe", "translate", "generate", "align", "reconstruct", "mix", "video"]`.
6. `test_edit_media_speech_wav` — write `media/speech.wav`; assert `["diarize", "samples", "transcribe", "translate", "generate", "align", "reconstruct", "mix", "video"]`.
7. `test_cli_from_stage_generate` — call with `from_stage="generate"`; assert `["generate", "align", "reconstruct", "mix", "video"]`.
8. `test_cli_force` — call with `force=True`; assert `list(STAGES)`.

Defensive: `test_no_changes_means_no_stale`, `test_missing_file_treated_as_hash_mismatch`, `test_to_stage_caps_stale_set`.

- [ ] **Step 2: Run — expect failure** — expect import error.

- [ ] **Step 3: Implement `src/workspace/dag.py`** — exactly per spec §10:
  * `STAGE_ORDER = ["extract","separate","diarize","samples","transcribe","translate","generate","align","reconstruct","mix","video"]`.
  * `@dataclass(frozen=True) CliOverrides(force=False, from_stage=None, to_stage=None)`.
  * `_is_editable(path, manifest)`: literal match or `*`-glob match in `manifest.editable_paths`.
  * `_is_derived(path, manifest)`: same against `derived_paths`.
  * `_stage_has_config_change(stage, manifest, recorded_configs)`: returns `stage in recorded_configs or False`; if `recorded_configs is None`, returns False.
  * `compute_stale_set(manifest, workspace_root, cli_overrides=None, *, recorded_configs=None) → list[str]`:
    1. Compute actual hashes for every path that exists in the workspace root.
    2. For each stage's inputs, if current != recorded: editable → add stage to stale; else → add producer + stage to stale.
    3. Add config-stale stages.
    4. Propagate downstream (BFS until fixed point): if a stage's non-derived input's producer is stale, the stage becomes stale.
    5. Apply CLI overrides (force → all; from_stage → STAGE_ORDER[idx:]; to_stage → cap).
    6. Return `[s for s in STAGE_ORDER if s in stale]`.

- [ ] **Step 4: Run — expect green** — expect 11 passed (8 spec rows + 3 defensive).

- [ ] **Step 5: Commit** — `git add src/workspace/dag.py tests/test_workspace_dag.py && git commit -m "feat(workspace): invalidation DAG (release blocker, spec §10)"`.

---

# Task 8: Class attributes + `subdir` arg on all 11 stages

**Files:** all 11 stage files; `tests/test_workspace_stage_metadata.py`.

The class attributes to add (use these exact values):

| Stage | inputs | outputs | editable | derived | config_fields |
|---|---|---|---|---|---|
| extract | `[]` | `[media/original_audio.wav, media_info.json]` | `[]` | `[]` | `[sample_rate]` |
| separate | `[media/original_audio.wav]` | `[media/speech.wav, media/background.wav]` | `[]` | `[]` | `[model, device, out_dir_name]` |
| diarize | `[media/speech.wav]` | `diarization/{segments.json,embeddings.npz,embeddings.meta.json,metadata.json}` | `[]` | `[diarization/embeddings.npz]` | `[model_id, min_speakers, max_speakers, no_pyannote]` |
| samples | `[media/speech.wav, diarization/segments.json]` | `speakers/speaker_01/{primary.wav,primary.txt,embedding.npy,metadata.json}` + `speakers/speaker_01/candidates/candidate_01.{wav,txt,score.json}` | `speakers/speaker_01/{primary.wav,primary.txt}` | `[]` | `[target_seconds, max_seconds]` |
| transcribe | `[media/speech.wav, diarization/segments.json]` | `transcription/{transcript.json,word_level.json}` | `[transcription/transcript.json]` | `[]` | `[model_size, source_language, device]` |
| translate | `[transcription/transcript.json]` | `translation/{translated_transcript.json,glossary.json}` | `translation/{translated_transcript.json,glossary.json}` | `[]` | `[source_language, target_language, backend_name]` |
| generate | `[translation/translated_transcript.json]` | `generated_segments/manifest.json` | `[]` | `[generated_segments/manifest.json]` | `[model_id, target_language, use_clone_prompt]` |
| align | `[generated_segments/manifest.json]` | `aligned_manifest.json` | `[]` | `[aligned_manifest.json]` | `[target_language, tolerance, regenerate_with_duration]` |
| reconstruct | `[aligned_manifest.json, media/background.wav]` | `output/reconstructed_speech.wav` | `[]` | `[output/reconstructed_speech.wav]` | `[target_sr]` |
| mix | `[output/reconstructed_speech.wav, media/background.wav]` | `output/final_audio.wav` | `[]` | `[output/final_audio.wav]` | `[target_lufs, speech_db, background_db]` |
| video | `[output/final_audio.wav]` | `output/final_video.mp4` | `[]` | `[output/final_video.mp4]` | `[]` |

- [ ] **Step 1: Write `tests/test_workspace_stage_metadata.py`** — `@pytest.mark.parametrize` over all 11 classes; each param case checks every attribute matches the table above. Plus a final test `test_editable_outputs_listed_in_editable_paths` that asserts the union of `editable_outputs` across stages is a superset of `{transcription/transcript.json, translation/translated_transcript.json, translation/glossary.json, speakers/speaker_01/primary.wav, speakers/speaker_01/primary.txt}`.

- [ ] **Step 2: Run — expect failure** — expect the parametrize tests to fail because the class attributes don't exist yet.

- [ ] **Step 3: Patch each of the 11 stage files** — for each:
  1. Add the 5 class attributes shown above.
  2. Add `subdir: str | None = None` keyword to `__init__`. When set, store `self.subdir` and reassign `self.workdir = self.workdir / subdir` **after** `Path(workdir)`. The legacy `Pipeline` callers pass no `subdir`, so they get the flat layout. The `WorkspacePipeline` passes the right `subdir` per stage.
  3. Stage-to-subdir mapping (used by WorkspacePipeline, **not** by the stage itself):
     * extract → `media`
     * separate → `media`
     * diarize → `diarization`
     * samples → `None` (outputs already namespaced as `speakers/...`)
     * transcribe → `transcription`
     * translate → `translation`
     * generate → `None` (outputs already `generated_segments/...`)
     * align → `None`
     * reconstruct → `None`
     * mix → `None`
     * video → `None`

  4. Do **not** change the body of `run()` — the existing `self.workdir / "x.wav"` lookups continue to work because the stage's `self.workdir` is already the nested subdir (when `subdir` was set). The flat layout (no subdir) also works as before.

- [ ] **Step 4: Run — expect green** — `.venv/bin/python -m pytest tests/test_workspace_stage_metadata.py -v`. Expect 12 passed (11 stages + final test).

- [ ] **Step 5: Run the existing test suite to ensure no regression** — `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_workspace_e2e.py --ignore=tests/test_workspace_edit_scenarios.py`. Expect everything green.

- [ ] **Step 6: Commit** — `git add src/stages/ tests/test_workspace_stage_metadata.py && git commit -m "refactor(stages): add inputs/outputs/config_fields class attrs + subdir arg"`.

---

# Task 9: `pipeline_config_dict()` on `src/pipeline.py`

**Files:** `src/pipeline.py`; `tests/test_pipeline_config_dict.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_pipeline_config_dict.py`:
  * `test_pipeline_config_dict_excludes_hf_token` — `Pipeline(input_path="/tmp/x", source_language="en", target_language="es", workdir=Path("/tmp"), hf_token="secret").pipeline_config_dict()` → `cfg["hf_token_available"] is True` and `"hf_token" not in cfg`.
  * `test_pipeline_config_dict_no_hf_token_marks_unavailable` — same call without `hf_token` → `cfg["hf_token_available"] is False`.
  * `test_pipeline_config_dict_includes_all_configurable_knobs` — `Pipeline(..., target_lufs=-14.0, no_pyannote=True, max_speakers=4).pipeline_config_dict()` → assert those values.

- [ ] **Step 2: Run — expect failure** — expect `AttributeError`.

- [ ] **Step 3: Add the method to `src/pipeline.py`** — define `def pipeline_config_dict(self) -> dict` returning:
  ```python
  {
      "whisper_model": self.whisper_model,
      "target_lufs": self.target_lufs,
      "no_pyannote": self.no_pyannote,
      "min_speakers": self.min_speakers,
      "max_speakers": self.max_speakers,
      "skip_video": self.skip_video,
      "glossary_path": str(self.glossary_path) if self.glossary_path else None,
      "hf_token_available": bool(self.hf_token),
  }
  ```
  Place it just below `__init__`.

- [ ] **Step 4: Run — expect green** — expect 3 passed.

- [ ] **Step 5: Commit** — `git add src/pipeline.py tests/test_pipeline_config_dict.py && git commit -m "feat(pipeline): pipeline_config_dict() excludes hf_token"`.

---

# Task 10: `src/workspace/pipeline.py` (WorkspacePipeline)

**Files:** `src/workspace/pipeline.py`; `tests/test_workspace_pipeline_helpers.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_workspace_pipeline_helpers.py`:
  * `test_stage_subdir` — `stage_subdir("extract") == "media"`, `("diarize") == "diarization"`, `("transcribe") == "transcription"`, `("translate") == "translation"`.
  * `test_stage_subdir_root_for_synthesis_stages` — `stage_subdir(s) is None` for `s in ("generate", "align", "reconstruct", "mix", "video")`.
  * `test_workspace_layout_lists_all_dirs` — `workspace_layout()` includes `media`, `diarization`, `transcription`, `translation`, `speakers`, `output`, `source`, `logs`.
  * `test_prepare_stages_stops_at_translate` — `prepare_stages() == ["extract","separate","diarize","samples","transcribe","translate"]`.
  * `test_generate_stages_starts_at_generate` — `generate_stages() == ["generate","align","reconstruct","mix","video"]`.

- [ ] **Step 2: Run — expect failure** — expect import error.

- [ ] **Step 3: Implement `src/workspace/pipeline.py`** — key points:
  * `STAGE_CLASSES = {"extract": ExtractStage, ..., "video": VideoStage}`.
  * `STAGE_SUBDIR = {"extract": "media", "separate": "media", "diarize": "diarization", "samples": None, "transcribe": "transcription", "translate": "translation", "generate": None, "align": None, "reconstruct": None, "mix": None, "video": None}`.
  * `stage_subdir(name) → Optional[str]` — dict lookup.
  * `prepare_stages() → list[str]` — `STAGE_ORDER[:STAGE_ORDER.index("translate")+1]`.
  * `generate_stages() → list[str]` — `STAGE_ORDER[STAGE_ORDER.index("generate"):]`.
  * `workspace_layout() → list[str]` — `[media, diarization, transcription, translation, speakers, output, source, logs]`.
  * `_build_stage(name, workspace_root, *, no_pyannote, hf_token, whisper_model, source_language, target_language, min_speakers, max_speakers, glossary_path, target_lufs)` — instantiates the right stage class with the right constructor kwargs (see Task 8 for the signature changes) and `subdir=STAGE_SUBDIR[name]`.
  * `@dataclass WorkspaceContext(workspace_root: Path, input_path: str, source_language: str, target_language: str, workdir="", output_dir="", _data={})` with `__post_init__` setting `self.workdir=str(self.workspace_root)` and `self.output_dir=str(self.workspace_root/"output")`, plus `__getitem__/__setitem__/__contains__/get` delegating to `self._data`.
  * `class WorkspacePipeline` with:
    * `__init__(self, input_path, source_language, target_language, *, whisper_model="large-v3", hf_token=None, target_lufs=-16.0, glossary_path=None, no_pyannote=False, min_speakers=None, max_speakers=None, workspace_root=None, output_dir=None)`.
    * `_config() → dict` (same fields as `Pipeline.pipeline_config_dict()`).
    * `_media_hash() → str` (delegate to `src.utils.cache.media_hash`).
    * `_id() → tuple[str, str, str]` returning `(workspace_id, content_hash, pipeline_config_hash)`.
    * `_ensure_layout(root) → None` (mkdir all dirs from `workspace_layout()`).
    * `prepare() → tuple[workspace_id, root]`:
      1. Compute workspace_id, content_hash, config_hash.
      2. `root = workspaces_root() / workspace_id`; `root.mkdir(parents=True, exist_ok=True)`.
      3. `_ensure_layout(root)`.
      4. Symlink `source/<input-name>` → original input (best-effort `try/except OSError`).
      5. Write `metadata.json` if missing (spec §8 schema).
      6. `_load_or_init_manifest(workspace_id)`.
      7. `_run_stages(manifest, prepare_stages(), start_at="extract")`.
      8. Return `(workspace_id, root)`.
    * `generate(workspace_id_str=None, *, from_stage=None, to_stage=None, force=False) → tuple[workspace_id, root]`:
      1. Resolve workspace_id (use arg or recompute via `_id()`).
      2. Load manifest from `<root>/manifest.json`.
      3. `stale = compute_stale_set(manifest, root, CliOverrides(force, from_stage, to_stage))`.
      4. If `stale` empty, log and return early.
      5. `_run_stages(manifest, stale, start_at=stale[0])`.
    * `_load_or_init_manifest(workspace_id, root=None)` — load existing or create new with the editable_paths from spec §9 and the union of all stages' `editable_outputs` / `derived_outputs`.
    * `_run_stages(manifest, stages, *, start_at)`:
      1. Build a `WorkspaceContext(workspace_root, input_path, ...)`.
      2. Pre-populate context keys: `audio_path`, `speech_path`, `background_path`, `segments_path`, `transcript_path`, `translated_path`, `generated_dir`, `manifest_path` (twice — first to `generated_segments/manifest.json` then overridden to `aligned_manifest` for downstream stages), `aligned_dir`, `aligned_manifest`, `reconstructed_path`, `final_path`, `final_video`.
      3. Skip stages before `start_at` (in case `stages[0] != start_at` because of `to_stage`).
      4. For each remaining stage, call `_run_single_stage`.
    * `_run_single_stage(manifest, name, context)`:
      1. `staging = stage_staging_dir(workspace_root, name); staging.mkdir(parents=True, exist_ok=True)`.
      2. Build a stage whose `workdir = staging` (so it writes into staging).
      3. Special-case `extract.run(self.input_path, context._data)`; others call `stage.run(context._data)`.
      4. On exception: `shutil.rmtree(staging)`; record `StageRecord(status="failed", error=...)` in manifest; re-raise.
      5. `promote(staging, workspace_root)`.
      6. Record `StageRecord(status="done", started_at, finished_at, duration_s, config={k: getattr(self, k) for k in stage_cls.config_fields}, inputs=[...], outputs=[...])` — for each declared input/output, if the file exists, append an `ArtifactRef(path, sha256, size_bytes)`.
      7. `manifest.save(workspace_root / "manifest.json")`.

- [ ] **Step 4: Run — expect green** — expect 5 passed.

- [ ] **Step 5: Commit** — `git add src/workspace/pipeline.py tests/test_workspace_pipeline_helpers.py && git commit -m "feat(workspace): WorkspacePipeline (prepare/generate) wrapping Pipeline"`.

---

# Task 11: `src/workspace/cli.py` (workspace subcommand handlers)

**Files:** `src/workspace/cli.py`; `tests/test_workspace_cli.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_workspace_cli.py`. Uses a helper `_seed_workspace(workspaces: Path) → wid` that materialises a minimal workspace (manifest.json + metadata.json + `media/original_audio.wav`). Tests:
  * `test_workspace_list` — `cmd_workspace_list(None)` prints workspace ID.
  * `test_workspace_inspect` — `cmd_workspace_inspect(_ns(workspace_id=wid))` prints "extract".
  * `test_workspace_show_prints_path` — `cmd_workspace_show(_ns(workspace_id=wid))` prints "media".
  * `test_workspace_validate_clean` — `cmd_workspace_validate(_ns(workspace_id=wid))` returns 0 and prints "no issues".
  * `test_workspace_clean_removes` — `cmd_workspace_clean(_ns(workspace_id=wid, keep_outputs=False, yes=True))` removes the workspace.

- [ ] **Step 2: Run — expect failure** — expect import error.

- [ ] **Step 3: Implement `src/workspace/cli.py`** — key points:
  * `_workspaces() → Path` (returns `workspaces_root()`).
  * `_all_workspaces() → list[Path]` (sorted, exists-filtered).
  * `cmd_workspace_list(args=None) → int` — table of `WORKSPACE ID / SRC / TGT / CREATED / SOURCE`; fall back to "No workspaces found." if empty.
  * `cmd_workspace_inspect(args) → int` — print header + stages with `status` and `duration_s`.
  * `cmd_workspace_show(args) → int` — print the path of a subdir (if `args.path`) or list all standard subdirs.
  * `_gather_issues(root) → list[Issue]` — call `validate_metadata`, `validate_manifest`, `validate_diarization_segments`, `validate_transcript`, `validate_translation`, `validate_glossary` on the relevant files (use `_try_load` to silently return `{}` for missing/broken).
  * `cmd_workspace_validate(args) → int` — print issues; return 2 if any error else 0.
  * `cmd_workspace_clean(args) → int` — confirm with `input("...")` unless `args.yes`; `shutil.rmtree(root)` (or just `output/` if `keep_outputs`).
  * `cmd_workspace_open(args) → int` — print root.

- [ ] **Step 4: Run — expect green** — expect 5 passed.

- [ ] **Step 5: Commit** — `git add src/workspace/cli.py tests/test_workspace_cli.py && git commit -m "feat(workspace): subcommand handlers (list/inspect/show/validate/clean/open)"`.

---

# Task 12: Extend `src/cli.py` with `prepare`, `generate`, `workspace`

**Files:** `src/cli.py`; `tests/test_cli_subcommands.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli_subcommands.py`:
  * `test_parser_has_prepare_generate_workspace()` — `build_parser().parse_args(["prepare","--input","x","--source-language","en","--target-language","es"]).command == "prepare"`. Similarly for `generate <wid>` and `workspace list`.

- [ ] **Step 2: Run — expect failure** — expect `SystemExit: 2`.

- [ ] **Step 3: Patch `src/cli.py`** — key edits:
  1. Add imports: `from .workspace.cli import (cmd_workspace_clean, cmd_workspace_inspect, cmd_workspace_list, cmd_workspace_open, cmd_workspace_show, cmd_workspace_validate)`, `from .workspace.pipeline import WorkspacePipeline`, `from .workspace.paths import workspaces_root`.
  2. In `build_parser`, add three new subparsers before the existing `cache` parser:
     * `prepare`: `--input/-i` (required), `--source-language/-s` (required), `--target-language/-t` (required), `--name`, `--whisper-model` (default "large-v3"), `--hf-token` (default `os.environ.get("HF_TOKEN")`), `--glossary`, `--no-pyannote` (action="store_true"), `--min-speakers` (int, default None), `--max-speakers` (int, default None).
     * `generate`: `workspace_id` (positional, nargs="?"), `--from-stage`, `--to-stage`, `--force` (action="store_true"), `--output-dir/-o` (default "output").
     * `workspace`: with subparsers `list`, `inspect <wid>`, `show <wid> [path]`, `open <wid>`, `validate <wid>`, `clean <wid> [--keep-outputs] [--yes]`.
  3. Add three handler functions and a dispatcher in `main`:
     * `_handle_prepare(args) → int` — check `Path(args.input).exists()`; build `WorkspacePipeline(...)`; call `wsp.prepare()`; print `[ok] Workspace ready: <id>` and `Path: <root>`.
     * `_handle_generate(args) → int` — if `args.workspace_id is None`, find the most recently modified workspace in `workspaces_root()`; build `WorkspacePipeline` with placeholder `input_path="/dev/null"`; call `wsp.generate(wid, from_stage=..., to_stage=..., force=...)`; print `[ok] Generated: <root>/output`.
     * `_handle_workspace(args) → int` — dispatch to the six `cmd_workspace_*` functions.
  4. In `main`, extend the pre-parser argv injection: `argv[0]` may now be `run|cache|glossary|prepare|generate|workspace|-h|--help`. Add the new branches in the if/elif chain.

- [ ] **Step 4: Run — expect green** — expect 1 passed.

- [ ] **Step 5: Run the full test suite (skip e2e/edit_scenarios for now)** — `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_workspace_e2e.py --ignore=tests/test_workspace_edit_scenarios.py`. Expect everything green.

- [ ] **Step 6: Commit** — `git add src/cli.py tests/test_cli_subcommands.py && git commit -m "feat(cli): add prepare/generate/workspace subcommands"`.

---

# Task 13: Update `dub.sh`

**Files:** `dub.sh`.

- [ ] **Step 1: Add new subcommand dispatchers** — after the `glossary` dispatcher (~line 108), add three `if [[ "${1:-}" == "prepare" ]]; then ... fi` / `generate` / `workspace` blocks that call `main.py <subcommand> "${@:2}"`.

- [ ] **Step 2: Print workspace ID on `run`** — after the existing `[ok] Wrote:` block at the end of `dub.sh`, append:
  ```bash
  WS_ROOT="${HOME}/.local/share/ai-dubbing/workspaces"
  if [[ -d "$WS_ROOT" ]]; then
      LATEST_WS=$(ls -1t "$WS_ROOT" 2>/dev/null | head -1 || true)
      if [[ -n "$LATEST_WS" ]]; then
          echo "     Workspace: $LATEST_WS  ($WS_ROOT/$LATEST_WS)"
      fi
  fi
  ```

- [ ] **Step 3: Update `print_usage`** — add the new commands and `--name` flag to the usage text (see Task 13 Step 1 in the original plan for the full text; it's compact enough to inline here).

- [ ] **Step 4: Verify syntax** — `bash -n dub.sh && echo OK`. Expect `OK`.

- [ ] **Step 5: Verify help** — `./dub.sh help`. Expect usage to include `prepare`, `generate`, `workspace`.

- [ ] **Step 6: Commit** — `git add dub.sh && git commit -m "feat(dub.sh): add prepare/generate/workspace commands; print workspace ID on run"`.

---

# Task 14: Integration tests

**Files:** `tests/test_workspace_e2e.py`, `tests/test_workspace_edit_scenarios.py`.

- [ ] **Step 1: Write `tests/test_workspace_e2e.py`** — uses `monkeypatch` to set `AI_DUBBING_WORKSPACES_ROOT=tmp_path`. Calls `WorkspacePipeline(...).prepare()` with a `input_path` set to the test fixture `tests/fixtures/short_sample.wav`. Because the stages will need pyannote/whisper/etc., monkey-patch `STAGE_CLASSES["extract"]` etc. with minimal in-process stubs that produce the file artefacts the manifest expects. Verify: workspace dir created; `metadata.json` has all required keys; `manifest.json` has all 11 stages with `status == "done"`; each declared output file exists; `validate_workspace(root)` returns no errors.

- [ ] **Step 2: Write `tests/test_workspace_edit_scenarios.py`** — uses the full manifest factory from `test_workspace_dag.py`. For each of the 8 §10 rows, build a workspace on disk, run `WorkspacePipeline.generate(workspace_id)`, then assert the right set of stages re-ran (use a recording stub that records which stages were invoked; reuse the stubs from e2e).

- [ ] **Step 3: Run the new tests** — `.venv/bin/python -m pytest tests/test_workspace_e2e.py tests/test_workspace_edit_scenarios.py -v`. Expect all green.

- [ ] **Step 4: Run the full suite** — `.venv/bin/python -m pytest tests/ -v`. Expect everything green.

- [ ] **Step 5: Commit** — `git add tests/test_workspace_e2e.py tests/test_workspace_edit_scenarios.py && git commit -m "test: end-to-end + edit-scenario integration tests"`.

---

# Task 15: `docs/workspaces.md` + `README.md`

**Files:** `docs/workspaces.md` (new), `README.md` (modify).

- [ ] **Step 1: Write `docs/workspaces.md`** — user guide covering:
  * What a workspace is (spec §1).
  * Workspace location (`~/.local/share/ai-dubbing/workspaces/<slug>-<YYYYMMDD>-<hash8>/`).
  * `dub.sh prepare <input> <src> <tgt>` — what it does, when to use it.
  * `dub.sh generate <workspace-id>` — what it does, `--from-stage`, `--to-stage`, `--force`.
  * `dub.sh workspace list` / `inspect` / `show` / `validate` / `clean`.
  * The 5 user-editable artefact paths and what they do (`translation/translated_transcript.json`, `translation/glossary.json`, `transcription/transcript.json`, `speakers/<id>/primary.wav`, `speakers/<id>/primary.txt`).
  * The three artifact classes (editable / non-editable / derived) and why they matter.
  * Examples of common workflows (fix a bad translation, swap a speaker sample, add a glossary entry).
  * Env vars (`AI_DUBBING_WORKSPACES_ROOT`, `XDG_DATA_HOME`).

- [ ] **Step 2: Update `README.md`** — add a "Workspaces" section linking to `docs/workspaces.md`; update the Quick Start to mention the new commands; add a "Two-phase workflow" bullet under Pipeline.

- [ ] **Step 3: Commit** — `git add docs/workspaces.md README.md && git commit -m "docs: workspace user guide + README quick start"`.

---

# Task 16: End-to-end smoke test

This task has no automated tests — it is a manual verification against the spec's mandatory smoke test (Deliverable #3).

- [ ] **Step 1: Run `prepare`** — `cd /home/bruno/github/tests/ai-dubbing && ./dub.sh prepare tests/fixtures/short_sample.wav pt en 2>&1 | tee /tmp/dub_prepare.log`. Expect: pipeline runs through extract..translate; prints a workspace ID and path; no errors. **Save the printed workspace ID** (e.g. `echo $WS_ID`).

- [ ] **Step 2: Run `generate`** — `./dub.sh generate $WS_ID 2>&1 | tee /tmp/dub_generate.log`. Expect: pipeline runs through generate..video; prints `final_audio.wav` (and `final_video.mp4` if the source was a video); no errors.

- [ ] **Step 3: Edit `translated_transcript.json`** — modify `$WS_ROOT/translation/translated_transcript.json` (e.g. change a `text` field). Re-run `./dub.sh generate $WS_ID 2>&1 | tee /tmp/dub_regen.log`. Verify that the log shows **only** `generate..video` re-running (no extract/separate/diarize/samples/transcribe/translate messages). The cheapest way: `grep -E "^\[(.*)\] (Audio Extraction|Demucs|Pyannote|Sample|Whisper|Translation|OmniVoice|Alignment|Reconstruction|Final Mix|Optional Video)" /tmp/dub_regen.log` should show ONLY the post-translate stages.

- [ ] **Step 4: Run `dub.sh run` for backwards-compat** — `cp tests/fixtures/short_sample.wav /tmp/test.wav && ./dub.sh /tmp/test.wav pt en /tmp/out.wav 2>&1 | tee /tmp/dub_run.log`. Expect: pipeline succeeds; the new `Workspace:` line is printed at the end; output file is created.

- [ ] **Step 5: Run `dub.sh cache list`** — `./dub.sh cache list`. Expect: legacy cache entries still listed (no migration required, per spec §16).

- [ ] **Step 6: Run pytest one more time** — `.venv/bin/python -m pytest tests/ -v`. Expect: all tests green.

- [ ] **Step 7: Commit any final fixes** — if Step 1–6 surfaced any issues, fix and commit them now. Conventional commit messages.

---

# Stopping criteria

Stop only when:

* All 16 tasks have commits.
* `pytest tests/ -v` is fully green.
* The smoke test in Task 16 (steps 1–3) passes: prepare → generate → edit → regenerate re-runs only `generate..video`.
* `dub.sh run` still works and prints the new workspace-ID line.
