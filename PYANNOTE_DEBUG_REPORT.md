# Pyannote Diarization Debug Report

## 1. Root cause

The Pyannote diarization failure had two distinct layers:

1. **Primary cause (account-level):** the HF account behind
   `HF_TOKEN` (`brunos3d`) had not accepted the gating terms for
   `pyannote/segmentation-3.0`.  Hugging Face responded with
   `huggingface_hub.errors.GatedRepoError: 403 Client Error` —
   *"Access to model pyannote/segmentation-3.0 is restricted and you
   are not in the authorized list."*  This blocked the entire
   pipeline because `pyannote/speaker-diarization-3.1` calls
   `Model.from_pretrained("pyannote/segmentation-3.0", ...)` internally.

2. **Secondary cause (code-level):** even after the account was
   granted access, the pipeline failed with
   `AttributeError: 'DiarizeOutput' object has no attribute
   'itertracks'`.  pyannote.audio **4.x** changed the return type of
   `Pipeline.__call__` from `pyannote.core.annotation.Annotation`
   to a new `DiarizeOutput` dataclass whose
   `.exclusive_speaker_diarization` attribute holds the
   `Annotation` we used to receive directly.

3. **Tertiary cause (UX-level):** the original code silently fell
   back to VAD+MFCC clustering whenever pyannote failed, hiding the
   real error and producing the *1 speaker / 14 segments* output
   reported in the bug.

All three causes have been addressed; pyannote now runs end-to-end on
`input/reunião.mp4` and produces a multi-speaker segmentation.

## 2. Authorisation findings

Investigation steps:

1. Loaded `HF_TOKEN` from `.env` via `python-dotenv`'s `load_dotenv`.
   Resolves to a 37-character token prefixed with `hf_`.
2. Confirmed the token is reaching Python: the diagnostics emitted by
   `src/stages/diarize.py` now print
   `HF_TOKEN loaded: True (hf_...LXX (len=37))`.
3. Confirmed the token is reaching `Pipeline.from_pretrained`:
   `_make_pipeline()` first tries the modern `token=` kwarg
   (pyannote.audio 4.x), falling back to the legacy `use_auth_token=`
   kwarg (3.x) via a `TypeError` catch.  Inspecting the signature
   confirms the rename:
   ```
   from_pretrained(checkpoint, revision=None, hparams_file=None,
                   token=None, cache_dir=None)
   ```
4. Ran `huggingface_hub.whoami()` against the token: returns
   `brunos3d` with a `read`-role access token.
5. Ran the decisive probe `huggingface_hub.hf_hub_download()`:

   | repo                                         | file              | result (before)    | result (after)   |
   |----------------------------------------------|-------------------|--------------------|------------------|
   | `pyannote/speaker-diarization-3.1`           | `config.yaml`     | PASS               | PASS             |
   | `pyannote/segmentation-3.0`                  | `config.yaml`     | FAIL (403)         | PASS             |
   | `pyannote/segmentation-3.0`                  | `pytorch_model.bin` | FAIL (403)       | PASS             |

6. After accepting the gating terms on
   https://huggingface.co/pyannote/segmentation-3.0 and
   https://huggingface.co/pyannote/speaker-diarization-3.1, all
   probes pass and `scripts/test_pyannote_auth.py` prints
   `RESULT: PASS - all required models reachable.`

## 3. Dependency findings

| package           | installed                | comment |
|-------------------|--------------------------|---------|
| torch             | 2.8.0+cu128              | matches `pyproject.toml` pin. |
| torchaudio        | 2.8.0+cu128              | matches `pyproject.toml` pin. |
| pyannote.audio    | 4.0.4                    | newer than the `>=3.1` floor in `pyproject.toml`.  4.x removed `use_auth_token=` *and* changed the return type of `Pipeline.__call__`; both are handled below. |
| pyannote.pipeline | 4.0.0                    | matches pyannote.audio 4.x. |
| lightning         | 2.6.5                    | required by pyannote. |
| huggingface_hub   | 1.19.0                   | exposes the GatedRepoError seen in the logs. |
| torchcodec        | 0.14.0                   | `libtorchcodec_core8.so` is built for torch 2.8, but its C++ symbols reference `torch_from_blob`, which is **not exported by torch 2.8.0+cu128's `libtorch_cpu.so`**. |
| ffmpeg            | 8.1.1 (system)           | provides `libavutil.so.60`.  torchcodec 0.14 was built against ffmpeg 7 (`libavutil.so.59`) and reverts to older `.so` lookups when its own bundled lib can't be loaded. |

**Why the torchcodec warning is harmless in this pipeline.**  pyannote
emits a `UserWarning` at import time telling the user to either fix
torchcodec or pre-load audio as a `{"waveform": tensor, "sample_rate": int}`
dict.  `src/stages/diarize.py` does exactly that (see
`_annotation_to_segments` callers):

```python
audio_in = {
    "waveform": torch.from_numpy(mono).float(),
    "sample_rate": sr,
}
```

The audio is loaded with `soundfile` in `src/utils/audio.py::read_wav`
*before* any pyannote call, so the broken torchcodec decoder is never
asked to read a file.  No pinning is required to make the inference
path work; the wheel-level symbol mismatch only affects pyannote's
file-path-based input mode which we don't use.

If the warning is bothersome in logs, it can be suppressed with
`PYTHONWARNINGS=ignore::UserWarning:pyannote.audio.core.io` in the
environment.  We deliberately do **not** pin a specific `torchcodec`
version in `pyproject.toml` because every available wheel has a
compatibility problem of its own and none of them are on the
inference code path.

## 4. Code changes

### 4.1 `src/stages/diarize.py`

- New `_resolve_hf_token()` and `_log_token_diagnostics()` helpers.
  Emit `HF_TOKEN loaded: True (hf_...LXX (len=37))` and
  `Pyannote auth mode: token= keyword` at the top of the diarize
  stage, so token resolution is auditable from the run log.
- New `_make_pipeline()` prefers the modern `token=` kwarg and only
  falls back to `use_auth_token=` for pyannote.audio 3.x.  This
  removes the `TypeError: from_pretrained() got an unexpected
  keyword argument 'use_auth_token'` noise on 4.x and silences the
  subsequent confusing fallback message.
- **`_annotation_to_segments()` updated for pyannote 4.x.**  The
  pipeline now returns a `DiarizeOutput` dataclass; we transparently
  unwrap `.exclusive_speaker_diarization` (the no-overlap
  `Annotation`) before iterating, while still accepting the bare
  `Annotation` returned by 3.x.
- Inverted the default.  Pyannote is the default diarizer; opting
  out is now an explicit `--no-pyannote` flag passed all the way
  from `dub.sh` → `cli.py` → `pipeline.py` → `DiarizeStage`.
- On any pyannote failure the stage now **raises** instead of
  silently falling back:
  > `Pyannote diarization failed and no fallback is allowed (default behaviour).  Re-run with --no-pyannote to opt out of pyannote and use the VAD+MFCC clustering fallback, or fix the underlying error above.`
- When pyannote is unreachable because `HF_TOKEN` is missing the
  stage now raises a clear actionable error:
  > `HF_TOKEN is not set but pyannote is the default diarizer.  Either set HF_TOKEN in .env (...) or re-run with --no-pyannote to use the VAD+MFCC clustering fallback.`
- Added a `traceback` import so the original `GatedRepoError` is
  preserved in the chained exception when running with `-vv`.
- Added a `Detected speakers: N` log line so the speaker count is
  visible without reading `segments.json`.

### 4.2 `src/cli.py`

- New `--no-pyannote` boolean flag, plumbed through to `Pipeline`.

### 4.3 `src/pipeline.py`

- New `no_pyannote` constructor arg on `Pipeline`, forwarded to
  `DiarizeStage`.

### 4.4 `dub.sh`

- Introduced a `BOOL_FLAGS` array (`--no-pyannote --no-cache
  --audio-only --no-video --read-only-cache`) and an
  `is_bool_flag()` helper so boolean flags are never accidentally
  consumed as the value of a preceding value-flag and never
  misinterpreted as the explicit output path.  This was needed
  because `--no-pyannote` was being mis-parsed as the output path
  by the previous heuristic.

### 4.5 `pyproject.toml`

- Added an explicit note on `pyannote.audio` explaining the gating
  requirement and pointing users at `scripts/test_pyannote_auth.py`.
- Added `python-dotenv` as a direct dependency (it was already
  imported in `cli.py` but not declared).
- Documented the torchcodec situation in the `torch-cuda` optional
  dependency block.  No pin is added because every available
  torchcodec wheel has a different incompatibility and none of them
  are on the inference path.

### 4.6 `scripts/test_pyannote_auth.py` (new)

Standalone diagnostic that loads `HF_TOKEN` from `.env`, then runs:

1. `HfApi.whoami()` to confirm the token resolves to a real account.
2. `HfApi.model_info()` for `pyannote/speaker-diarization-3.1` and
   `pyannote/segmentation-3.0`.
3. `HfApi.hf_hub_download()` (decisive probe) for the same repos,
   attempting to fetch `config.yaml` and `pytorch_model.bin`.
4. `pyannote.audio.Pipeline.from_pretrained(..., token=...)` so the
  verifier exercises the same code path as the pipeline.

Prints `PASS:` / `FAIL:` per check and a final `RESULT: PASS` /
`RESULT: FAIL` summary.  Exits 0 on success, 1 on failure.

Run it directly:

```bash
.venv/bin/python scripts/test_pyannote_auth.py
```

## 5. Validation results

### 5.1 Auth verification (final, after gating accepted)

```
$ .venv/bin/python scripts/test_pyannote_auth.py
================================================================
Pyannote / Hugging Face access verification
================================================================
HF_TOKEN source : env
HF_TOKEN value  : hf_...LXX (len=37)
Python          : 3.12.13

--- Token reachability ---
PASS: whoami ok: brunos3d (Bruno Silva)

--- Model metadata ---
PASS: pyannote/speaker-diarization-3.1 -> model_info ok (gated=auto, sha=84fd2591)
PASS: pyannote/segmentation-3.0        -> model_info ok (gated=auto, sha=e66f3d3b)

--- Decisive download probe ---
PASS: pyannote/speaker-diarization-3.1 :: download ok: config.yaml
PASS: pyannote/segmentation-3.0        :: download ok: config.yaml
PASS: pyannote/segmentation-3.0        :: download ok: pytorch_model.bin

--- Pyannote pipeline load ---
PASS: Pipeline loaded: SpeakerDiarization
================================================================
RESULT: PASS - all required models reachable.
```

### 5.2 Pipeline run with strict default — successful

```
$ ./dub.sh input/reunião.mp4 pt en --no-cache
>> Input   : input/reunião.mp4
>> Source  : pt
>> Target  : en
>> Output  : ...
...
[INFO] HF_TOKEN loaded: True (hf_...LXX (len=37))
[INFO] Pyannote auth mode: token= keyword
[INFO] Detected speakers: 2
[INFO] Pyannote: 2 speakers / 19 segments
[INFO]   speaker_01: primary profile selected
[INFO]   speaker_02: primary profile selected
[INFO]   speaker_01: dur=11.06s score=83.9 [SELECTED]
[INFO]   speaker_02: dur=25.94s score=54.6 [SELECTED]
[INFO] Detected language: pt (prob 1.00)
[INFO] Segment   1: speaker=speaker_01 profile=primary
[INFO] Segment   2: speaker=speaker_02 profile=primary
[INFO] Segment   3: speaker=speaker_02 profile=primary
[INFO] Segment   4: speaker=speaker_02 profile=primary
[INFO] Segment   5: speaker=speaker_01 profile=primary
[INFO] Segment   6: speaker=speaker_01 profile=primary
[INFO] Segment   7: speaker=speaker_02 profile=primary
[INFO] Segment   8: speaker=speaker_01 profile=primary
[INFO] Segment   9: speaker=speaker_02 profile=primary
[INFO] Segment  10: speaker=speaker_01 profile=primary
[INFO] Segment  11: speaker=speaker_01 profile=primary
[INFO] Segment  12: speaker=speaker_01 profile=primary
[INFO] Segment  13: speaker=speaker_01 profile=primary
[INFO] Segment  14: speaker=speaker_01 profile=primary
[ok] Wrote: /home/bruno/github/tests/ai-dubbing/input/reunião-dub-en.mp4
```

The output file is 20.8 MB, 92.9 s duration, ~1.88 Mbit/s.  No
fallback path was taken; the only diarizer in use was pyannote.

### 5.3 Pipeline run with `--no-pyannote` (forced fallback) — still works

```
$ .venv/bin/python main.py run --input "input/reunião.mp4" \
    --source-language pt --target-language en --output-dir /tmp/dub-test \
    --no-cache --no-pyannote
...
[ok] Dubbing complete.
     Video : /tmp/dub-test/final_video.mp4
```

This confirms the VAD+MFCC fallback path is still available as an
opt-in escape hatch.

### 5.4 Speaker counts before and after

| run                                              | mode              | speakers            | segments |
|--------------------------------------------------|-------------------|---------------------|----------|
| previous log (no flag, old code)                 | fallback (silent) | 1                   | 14       |
| `... --no-cache --no-pyannote` (new code)        | fallback          | 1                   | 14       |
| `./dub.sh ... --no-cache` (default, new code)    | **pyannote**      | **2** (speaker_01 + speaker_02) | **19** |

The 1 → 2 speaker count and 14 → 19 segment count are the direct
result of switching from MFCC clustering (which can't tell the two
voices apart) to pyannote.audio 4.x (which can).

## 6. Is Pyannote executing successfully?

**Yes.**  All four required probes in `scripts/test_pyannote_auth.py`
pass, the pipeline runs end-to-end, the diarize stage logs
`Detected speakers: 2`, and the final dubbed video
(`input/reunião-dub-en.mp4`, 20.8 MB, 92.9 s) is produced without
ever entering the VAD+MFCC fallback.

The remaining torchcodec import-time warning is benign on this
inference path (audio is pre-loaded as a `{"waveform": tensor,
"sample_rate": int}` dict); it does not affect diarization quality
or pipeline success.

## 7. Summary of the commits

Two commits, both on `main`:

1. `d257830 fix: restore pyannote diarization and eliminate fallback authorization failures`
   - inverts the diarize default to pyannote
   - adds `--no-pyannote` opt-out
   - adds HF_TOKEN diagnostics and modern `token=` kwarg
   - adds `BOOL_FLAGS` handling in `dub.sh`
   - adds `scripts/test_pyannote_auth.py`
   - adds `PYANNOTE_DEBUG_REPORT.md`
2. (this report update) documents the final pyannote 4.x
   `DiarizeOutput` API fix in `_annotation_to_segments` and the
   successful end-to-end run.

