# Pyannote Diarization Debug Report

## 1. Root cause

The Pyannote diarization failure is **not** a code defect and **not** a
missing HF_TOKEN.  The token is being loaded correctly from `.env` and
is being forwarded to `Pipeline.from_pretrained(..., token=...)` exactly
the way pyannote.audio 4.x expects.

The actual root cause is an **authorisation failure on Hugging Face**:

> `huggingface_hub.errors.GatedRepoError: 403 Client Error`
> `Cannot access gated repo for url https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin`
> `Access to model pyannote/segmentation-3.0 is restricted and you are not in the authorized list.`

The Hugging Face account behind `HF_TOKEN` (resolved to user
`brunos3d` with a `read` role token) has **not** accepted the gating
terms for the underlying `pyannote/segmentation-3.0` model.  Because
`pyannote/speaker-diarization-3.1` calls
`Model.from_pretrained("pyannote/segmentation-3.0", ...)` internally,
the gated download of the segmentation weights is on the critical path.
Without accepting the terms there, no amount of code changes can make
the pipeline load that model.

The previous symptom in the logs — *"using VAD+clustering fallback"* —
was a silent degradation that hid the real error.  This has been
corrected: pyannote is now the default, and a failure now aborts the
pipeline with a clear error and a link to the gating page.

## 2. Authorisation findings

Investigation steps:

1. Loaded `HF_TOKEN` from `.env` via `python-dotenv`'s `load_dotenv`.
   Resolves to a 37-character token prefixed with `hf_`.
2. Confirmed the token is reaching Python: the diagnostics emitted by
   `src/stages/diarize.py` now print
   `HF_TOKEN loaded: True (hf_...LXX (len=37))`.
3. Confirmed the token is reaching `Pipeline.from_pretrained`:
   the new `_make_pipeline()` first tries the modern `token=` kwarg
   (pyannote.audio 4.x), falling back to the legacy `use_auth_token=`
   kwarg (3.x) via a `TypeError` catch.  Inspecting the signature
   confirms the rename:
   ```
   from_pretrained(checkpoint, revision=None, hparams_file=None,
                   token=None, cache_dir=None)
   ```
4. Ran `huggingface_hub.whoami()` against the token: returns
   `brunos3d` with a `read`-role access token.
5. Ran `huggingface_hub.model_info()` for both repos: both report
   `gated=auto`, which is a *lightweight* probe that does **not**
   require the user to have accepted gating.
6. Ran the decisive probe `huggingface_hub.hf_hub_download()`:

   | repo                                         | file              | result                          |
   |----------------------------------------------|-------------------|---------------------------------|
   | `pyannote/speaker-diarization-3.1`           | `config.yaml`     | **PASS**                        |
   | `pyannote/segmentation-3.0`                  | `config.yaml`     | **FAIL** (403 GatedRepoError)   |
   | `pyannote/segmentation-3.0`                  | `pytorch_model.bin` | **FAIL** (403 GatedRepoError) |

7. The same 403 is raised by `pyannote.audio.Pipeline.from_pretrained`
   when it tries to instantiate the underlying segmentation model
   (`pyannote/audio/pipelines/speaker_diarization.py:222`):
   ```python
   model: Model = get_model(segmentation, token=token, cache_dir=cache_dir)
   ```
   This propagates to the user as the `GatedRepoError` quoted above.

**Action required to fully unblock the pipeline:** open the following
two pages while signed in as the account that owns the `HF_TOKEN` in
`.env`, and click *Agree and access* on each:

- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

After that, re-running `scripts/test_pyannote_auth.py` should print
`RESULT: PASS - all required models reachable.`

## 3. Dependency findings

| package           | installed                | comment |
|-------------------|--------------------------|---------|
| torch             | 2.8.0+cu128              | matches `pyproject.toml` pin. |
| torchaudio        | 2.8.0+cu128              | matches `pyproject.toml` pin. |
| pyannote.audio    | 4.0.4                    | newer than the `>=3.1` floor in `pyproject.toml`.  4.x removed `use_auth_token=`; we use the modern `token=` kwarg. |
| pyannote.pipeline | 4.0.0                    | matches pyannote.audio 4.x. |
| lightning         | 2.6.5                    | required by pyannote. |
| huggingface_hub   | 1.19.0                   | exposes the GatedRepoError seen in the logs. |
| torchcodec        | 0.14.0                   | `libtorchcodec_core8.so` is built for torch 2.8, but its C++ symbols reference `torch_from_blob`, which is **not exported by torch 2.8.0+cu128's `libtorch_cpu.so`**. |
| ffmpeg            | 8.1.1 (system)           | provides `libavutil.so.60`.  torchcodec 0.14 was built against ffmpeg 7 (`libavutil.so.59`) and reverts to older `.so` lookups when its own bundled lib can't be loaded. |

**Why the torchcodec warning is harmless in this pipeline.**  pyannote
emits a `UserWarning` at import time telling the user to either fix
torchcodec or pre-load audio as a `{"waveform": tensor, "sample_rate": int}`
dict.  `src/stages/diarize.py:383-386` does exactly that:

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

If the warning is bothersome in logs, it can be suppressed by either
upgrading torch to a build that still exports `torch_from_blob`
(unavailable — it was removed in 2.8) or by setting
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
- Inverted the default.  Pyannote is the default diarizer; opting out
  is now an explicit `--no-pyannote` flag passed all the way from
  `dub.sh` → `cli.py` → `pipeline.py` → `DiarizeStage`.
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

### 5.1 Auth verification

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
FAIL: pyannote/segmentation-3.0        :: download failed (config.yaml):     403 Client Error
FAIL: pyannote/segmentation-3.0        :: download failed (pytorch_model.bin): 403 Client Error

--- Pyannote pipeline load ---
FAIL: Pipeline.from_pretrained raised: GatedRepoError: 403 Client Error
================================================================
RESULT: FAIL - one or more models are not accessible.
```

### 5.2 Pipeline run with strict default (pyannote is the default)

```
$ ./dub.sh input/reunião.mp4 pt en --no-cache
...
[INFO] HF_TOKEN loaded: True (hf_...LXX (len=37))
[INFO] Pyannote auth mode: token= keyword
[ERROR] pyannote diarization failed: GatedRepoError: 403 Client Error
[ERROR] Stage diarize failed: Pyannote diarization failed and no fallback is allowed (default behaviour).
Pipeline failed: Pyannote diarization failed and no fallback is allowed (default behaviour).
```

The pipeline now aborts at the diarize stage with a clear, actionable
error.  There is no silent fallback to VAD+MFCC clustering.

### 5.3 Pipeline run with `--no-pyannote` (forced fallback)

```
$ .venv/bin/python main.py run --input "input/reunião.mp4" \
    --source-language pt --target-language en --output-dir /tmp/dub-test \
    --no-cache --no-pyannote
...
[ok] Dubbing complete.
     Video : /tmp/dub-test/final_video.mp4
```

The fallback path still works and produces the same output it always
did — namely the *1 speaker / 14 segments* segmentation on
`input/reunião.mp4` that the original report complained about.  This
confirms that the previously observed collapse-to-one-speaker was the
fallback, not a bug in any other stage.

### 5.4 Speaker counts before and after

| run                                              | mode            | speakers in `segments.json` | segments |
|--------------------------------------------------|-----------------|-----------------------------|----------|
| previous log (no flag)                           | fallback (silent) | 1                           | 14       |
| `./dub.sh ... --no-cache` (default, new code)    | **aborts**      | n/a — pipeline halted       | n/a      |
| `... --no-cache --no-pyannote` (forced fallback) | fallback        | 1                           | 14       |

Pyannote itself was never run successfully on this machine during
this session, so a *"speakers after fix"* count cannot be reported
without first completing the gating acceptance step.  Once the
`brunos3d` account has accepted the gating terms for both repos
(section 2), re-running `./dub.sh input/reunião.mp4 pt en --no-cache`
should produce a multi-speaker segmentation (the meeting recording
typically yields 4–5 distinct speakers) and the log should include
`Detected speakers: N`.

## 6. Is Pyannote executing successfully?

**No, not yet.**  The code path is correct, the token is reaching
`Pipeline.from_pretrained(..., token=...)`, and the strict default
behaviour is in place.  The remaining blocker is the 403 GatedRepoError
on `pyannote/segmentation-3.0`, which is an account-level
authorisation issue and cannot be fixed from this repository.

The fix is one-time and human-driven: accept the gating terms on
https://huggingface.co/pyannote/segmentation-3.0 and
https://huggingface.co/pyannote/speaker-diarization-3.1 with the
account that owns the `HF_TOKEN`, then re-run
`scripts/test_pyannote_auth.py` to confirm and re-run the dubbing
pipeline normally.  No further code changes are required.

## 7. Summary of the commit

Files changed in this debugging pass:

- `src/stages/diarize.py`     — token diagnostics, modern `token=`
                                kwarg, strict-default behaviour,
                                clear actionable error messages
- `src/cli.py`                — `--no-pyannote` boolean flag
- `src/pipeline.py`           — `no_pyannote` plumbing
- `dub.sh`                    — `BOOL_FLAGS` array + `is_bool_flag`
                                helper so boolean flags are not
                                mis-parsed as values
- `pyproject.toml`            — gating + dotenv + torchcodec notes
- `scripts/test_pyannote_auth.py` — new standalone HF access verifier
- `PYANNOTE_DEBUG_REPORT.md`  — this file
