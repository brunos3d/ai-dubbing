#!/usr/bin/env bash
# dub.sh - ffmpeg-style multilingual dubbing wrapper
#
# Usage:
#   dub.sh <input> <source-language> <target-language> [output-file] [options]
#
# Examples:
#   dub.sh video.mp4 pt en                       # -> ./video-dub-en.mp4
#   dub.sh /data/in.mp4 pt en ~/dubbed.mp4      # -> ~/dubbed.mp4
#   dub.sh podcast.mp3 en es out.mp3            # -> ./out.mp3 (audio only)
#   dub.sh video.mp4 pt en -- --whisper-model=medium
#
# Output path resolution:
#   * 4th positional arg, if given, is the explicit output file path.
#   * Otherwise, output is written next to the input as
#     <stem>-dub-<target-language>.<ext>.
#   * If the resolved output already exists, the suffix -(1), -(2) ...
#     is appended automatically.  Existing files are never overwritten.
#
# Format selection is driven by the output extension:
#   .mp4 / .mkv / .mov / .webm / .m4v   -> full video
#   .wav / .mp3 / .flac / .m4a / .ogg    -> audio only
#   Override with --audio-only or --emit=audio|video|both.
#
# Anything after the literal `--` is forwarded to main.py verbatim.

set -euo pipefail

# Anchor to the project root (the directory this script lives in).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_usage() {
    cat <<'EOF'
ai-dubbing: Local-first multilingual dubbing pipeline

Usage:
  dub.sh <input> <src-lang> <tgt-lang> [output] [options]
  dub.sh prepare <input> <src> <tgt> [--name <slug>] [--glossary <file>]
  dub.sh generate <workspace-id> [--from-stage NAME] [--to-stage NAME] [--force]
  dub.sh workspace <command>
  dub.sh cache <command>
  dub.sh glossary <command>

Commands:
  run (default)      Execute the dubbing pipeline on an input file.
  prepare            Create a workspace and run stages up to translate.
  generate           Run a workspace from generate..video (resume/edit flow).
  workspace list     List all workspaces.
  workspace inspect  Show metadata + manifest summary for a workspace.
  workspace show     Print a file from inside a workspace.
  workspace validate Validate every declared artifact in a workspace.
  workspace clean    Delete a workspace (keeps outputs by default).
  cache info         Show aggregate cache statistics.
  cache list         List all cached pipeline runs.
  cache prune        Remove stale cache entries (>7 days old).
  cache clear        Remove all cached artifacts.
  glossary template  Generate a sample entity_glossary.json.

Primary Options:
  -i, --input        Path to input media (mp4/mkv/mov/mp3/wav/flac).
  -s, --src-lang     Source language code (e.g. en, pt, ru).
  -t, --tgt-lang     Target language code (e.g. en, pt-BR, es).
  -o, --output-dir   Where to write final outputs (default: output/).
  --name SLUG        Workspace slug override (prepare only).
  --glossary PATH    JSON file for preserving names/brands from translation.
  --no-cache         Ignore existing cache and rebuild from scratch.
  --from-stage NAME  Reuse cache up to NAME, then rebuild.
  --to-stage NAME    Stop after NAME (generate only).
  --force            Re-run all stages regardless of cache (generate only).
  --audio-only       Skip video remux; emit only final_audio.wav.

Manual & Tips for Professional Dubbing:

1. Named Entity Preservation:
   If names like "Ei Nerd" or "Marvel" are being translated, use a glossary:
   $ dub.sh glossary template
   $ [Edit entity_glossary.json with your terms]
   $ dub.sh input.mp4 pt en --glossary entity_glossary.json

2. Two-Phase Workflow (prepare / generate):
   $ dub.sh prepare  input.mp4 pt en --name myvideo
   $ dub.sh generate <workspace-id>          # run generate..video
   Edit files under the workspace root, then re-run `generate` to resume.

3. Improving Quality:
   - Use 'large-v3' (default) for transcription.
   - If audio is noisy, Demucs separation (automatic) will help.

Output detection:
  .mp4 / .mkv / .mov   full video
  .wav / .mp3 / .flac  audio only
  Override via --audio-only / --emit=audio|video|both
EOF
}

# ---------------------------------------------------------------- argparse --

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    print_usage
    exit 0
fi

# Handle cache subcommand
if [[ "${1:-}" == "cache" ]]; then
    PYTHON_BIN=".venv/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3)"
    fi
    "$PYTHON_BIN" main.py cache "${@:2}"
    exit 0
fi

# Handle glossary subcommand
if [[ "${1:-}" == "glossary" ]]; then
    PYTHON_BIN=".venv/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3)"
    fi
    "$PYTHON_BIN" main.py glossary "${@:2}"
    exit 0
fi

# Handle prepare subcommand
if [[ "${1:-}" == "prepare" ]]; then
    PYTHON_BIN=".venv/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3)"
    fi
    # If the first three args are not flags, assume they are <input> <src> <tgt>
    if [[ $# -ge 4 && "${2:-}" != -* && "${3:-}" != -* && "${4:-}" != -* ]]; then
        INPUT="$2"
        SRC="$3"
        TGT="$4"
        shift 4
        "$PYTHON_BIN" main.py prepare --input "$INPUT" --source-language "$SRC" --target-language "$TGT" "$@"
    else
        "$PYTHON_BIN" main.py prepare "${@:2}"
    fi
    exit 0
fi

# Handle generate subcommand
if [[ "${1:-}" == "generate" ]]; then
    PYTHON_BIN=".venv/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3)"
    fi
    "$PYTHON_BIN" main.py generate "${@:2}"
    exit 0
fi

# Handle workspace subcommand
if [[ "${1:-}" == "workspace" ]]; then
    PYTHON_BIN=".venv/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3)"
    fi
    "$PYTHON_BIN" main.py workspace "${@:2}"
    exit 0
fi

if [[ $# -lt 3 ]]; then
    print_usage
    exit 1
fi

INPUT="$1"
SRC="$2"
TGT="$3"
shift 3

# Split args at `--`: anything before is dub.sh's own, anything after is
# forwarded to main.py.
PRE_DASH=()
POST_DASH=()
SEEN_SEP=0
for arg in "$@"; do
    if [[ $SEEN_SEP -eq 0 && "$arg" == "--" ]]; then
        SEEN_SEP=1
        continue
    fi
    if [[ $SEEN_SEP -eq 1 ]]; then
        POST_DASH+=("$arg")
    else
        PRE_DASH+=("$arg")
    fi
done

# First non-flag pre-dash arg = explicit output path; the rest are flags
# we forward to main.py. Flags that take a value consume the next arg so
# it isn't mistaken for the output path.
EXPLICIT_OUTPUT=""
MAIN_FLAGS=()
SKIP_NEXT=0
# Flags that take a value (consumes the next arg as that value).
VALUE_FLAGS=(
    --glossary --adapt-model --adapt-mode --whisper-model --target-lufs
    --hf-token --workdir --output-dir --start-from --only --from-stage
    --emit -o
)
# Boolean flags forwarded to main.py without taking a value.
BOOL_FLAGS=(--no-pyannote --no-cache --audio-only --no-video --read-only-cache)
is_value_flag() {
    local flag="$1"
    for vf in "${VALUE_FLAGS[@]}"; do
        if [[ "$flag" == "$vf" ]]; then
            return 0
        fi
    done
    return 1
}
is_bool_flag() {
    local flag="$1"
    for bf in "${BOOL_FLAGS[@]}"; do
        if [[ "$flag" == "$bf" ]]; then
            return 0
        fi
    done
    return 1
}
for arg in "${PRE_DASH[@]+"${PRE_DASH[@]}"}"; do
    if [[ $SKIP_NEXT -eq 1 ]]; then
        MAIN_FLAGS+=("$arg")
        SKIP_NEXT=0
        continue
    fi
    if is_value_flag "$arg"; then
        MAIN_FLAGS+=("$arg")
        SKIP_NEXT=1
        continue
    fi
    if is_bool_flag "$arg"; then
        # Standalone boolean flag - never treats the next arg as its value
        # and is never misinterpreted as the explicit output path.
        MAIN_FLAGS+=("$arg")
        continue
    fi
    if [[ -z "$EXPLICIT_OUTPUT" && "$arg" != -* && -n "$arg" ]]; then
        EXPLICIT_OUTPUT="$arg"
    else
        MAIN_FLAGS+=("$arg")
    fi
done

# ----------------------------------------------- resolve default output --

if [[ -z "$EXPLICIT_OUTPUT" ]]; then
    # Default: <input-stem>-dub-<tgt>.<input-ext> in the same dir as the input.
    # Use python to be robust against paths with spaces / unicode / macOS /tmp.
    EXPLICIT_OUTPUT=$(python3 - "$INPUT" "$TGT" <<'PY'
import os, sys
inp, tgt = sys.argv[1], sys.argv[2]
d, base = os.path.split(os.path.abspath(inp))
stem, dot, ext = base.partition(".")
if not dot:
    stem, ext = base, "mp4"
# Sanitize target language token for filesystem
tgt_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tgt)
print(os.path.join(d, f"{stem}-dub-{tgt_safe}.{ext}"))
PY
)
fi

# Make the directory if it doesn't exist.
OUT_DIR=$(dirname -- "$EXPLICIT_OUTPUT")
mkdir -p -- "$OUT_DIR"

# Resolve symlinks in the parent dir so the collision check below
# is robust against e.g. /tmp -> /private/tmp on macOS.
if [[ -d "$OUT_DIR" ]]; then
    OUT_DIR=$(cd -- "$OUT_DIR" && pwd)
fi

OUT_STEM=$(basename -- "$EXPLICIT_OUTPUT" | sed -E 's/\.[^.]+$//')
OUT_EXT=$(basename -- "$EXPLICIT_OUTPUT" | sed -E 's/^[^.]*\.//')
OUT_EXT_LOWER=$(printf '%s' "$OUT_EXT" | tr '[:upper:]' '[:lower:]')

# If the user gave a path whose directory is relative, anchor it to
# the input's directory so the output lands next to the input by default.
if [[ "$OUT_DIR" != /* && "$OUT_DIR" != .* && "$OUT_DIR" != /* ]]; then
    IN_DIR=$(dirname -- "$(cd -- "$(dirname -- "$INPUT")" && pwd)/$(basename -- "$INPUT")")
    OUT_DIR="$IN_DIR/$OUT_DIR"
fi
# Re-anchor the explicit output to the resolved dir.
EXPLICIT_OUTPUT="$OUT_DIR/$OUT_STEM$([[ -z "$OUT_EXT" ]] || echo .)$OUT_EXT"

# --------------------------------------------- pick emit from extension --

case "$OUT_EXT_LOWER" in
    mp4|mkv|mov|webm|m4v) DEFAULT_EMIT="video" ;;
    wav|mp3|flac|m4a|ogg|aac) DEFAULT_EMIT="audio" ;;
    *) DEFAULT_EMIT="auto" ;;
esac

# ----------------------------------------------- find non-colliding path --

# Pre-compute absolute version of the requested output so we can compare
# reliably against the input.
ABS_EXPLICIT=$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$EXPLICIT_OUTPUT")
ABS_INPUT=$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$INPUT")

FINAL_PATH="$ABS_EXPLICIT"
COUNTER=1
while [[ -e "$FINAL_PATH" ]]; do
    FINAL_PATH="${OUT_DIR}/${OUT_STEM}-(${COUNTER})$([[ -z "$OUT_EXT" ]] || echo .)${OUT_EXT}"
    COUNTER=$((COUNTER + 1))
done

# Refuse to overwrite the input even if user explicitly points there.
if [[ "$FINAL_PATH" == "$ABS_INPUT" ]]; then
    FINAL_PATH="${OUT_DIR}/${OUT_STEM}-(${COUNTER})$([[ -z "$OUT_EXT" ]] || echo .)${OUT_EXT}"
fi

# ----------------------------------------------------- build main flags --

declare -a PYTHON_FLAGS=()
EMIT_VALUE=""
EMIT_NEXT=0
for flag in "${MAIN_FLAGS[@]+"${MAIN_FLAGS[@]}"}"; do
    case "$flag" in
        --audio-only) PYTHON_FLAGS+=("--audio-only") ;;
        --emit=*)     EMIT_VALUE="${flag#--emit=}" ;;
        --emit)       EMIT_NEXT=1 ;;
        *)
            if [[ $EMIT_NEXT -eq 1 ]]; then
                EMIT_VALUE="$flag"
                EMIT_NEXT=0
            else
                PYTHON_FLAGS+=("$flag")
            fi
            ;;
    esac
done

# Emit precedence: explicit --emit flag > --audio-only > extension default > auto.
# Track the EFFECTIVE emit (what the pipeline will actually produce) so the
# post-run copy step can pick the right artefact, even when --audio-only
# overrides an .mp4 extension heuristic.
if [[ -n "$EMIT_VALUE" ]]; then
    EFFECTIVE_EMIT="$EMIT_VALUE"
    PYTHON_FLAGS+=("--emit" "$EMIT_VALUE")
elif [[ " ${MAIN_FLAGS[*]+"${MAIN_FLAGS[*]}"} " == *" --audio-only "* ]]; then
    # --audio-only is an explicit user intent and wins over the extension
    # heuristic (e.g. .mp4 + --audio-only should still produce audio).
    EFFECTIVE_EMIT="audio"
    PYTHON_FLAGS+=("--emit" "audio")
elif [[ "$DEFAULT_EMIT" != "auto" ]]; then
    EFFECTIVE_EMIT="$DEFAULT_EMIT"
    PYTHON_FLAGS+=("--emit" "$DEFAULT_EMIT")
else
    EFFECTIVE_EMIT="auto"
fi
PYTHON_FLAGS+=("${POST_DASH[@]+"${POST_DASH[@]}"}")

# --------------------------------------------------------- run pipeline --

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

echo ">> Input   : $INPUT"
echo ">> Source  : $SRC"
echo ">> Target  : $TGT"
echo ">> Output  : $FINAL_PATH"
echo ">> Format  : $EFFECTIVE_EMIT (driven by .$OUT_EXT)"

# Use a temporary directory for the pipeline's internal output to avoid
# repository pollution.
INTERNAL_OUT=$(mktemp -d)
trap 'rm -rf "$INTERNAL_OUT"' EXIT

# Suppress the CLI's own success banner — dub.sh will print its own.
DUB_SHOW_BANNER=1 "$PYTHON_BIN" main.py \
    --input "$INPUT" \
    --source-language "$SRC" \
    --target-language "$TGT" \
    --output-dir "$INTERNAL_OUT" \
    "${PYTHON_FLAGS[@]}"

# ------------------------------------------------ deliver to final path --

# Decide which artefact the pipeline produced.
SRC_FILE=""
# 1. Prefer the format implied by the user's flag / effective emit.
case "$EFFECTIVE_EMIT" in
    audio)
        [[ -f "$INTERNAL_OUT/final_audio.wav" ]] && SRC_FILE="$INTERNAL_OUT/final_audio.wav"
        ;;
    *)
        if [[ -f "$INTERNAL_OUT/final_video.mp4" ]]; then
            SRC_FILE="$INTERNAL_OUT/final_video.mp4"
        elif [[ -f "$INTERNAL_OUT/final_audio.wav" ]]; then
            SRC_FILE="$INTERNAL_OUT/final_audio.wav"
        fi
        ;;
esac
# 2. Fall back: take whatever exists.
if [[ -z "$SRC_FILE" ]]; then
    for cand in "$INTERNAL_OUT/final_video.mp4" "$INTERNAL_OUT/final_audio.wav"; do
        if [[ -f "$cand" ]]; then SRC_FILE="$cand"; break; fi
    done
fi
if [[ -z "$SRC_FILE" ]]; then
    echo "ERROR: pipeline produced no output file in $INTERNAL_OUT" >&2
    exit 1
fi

SRC_EXT_LOWER=$(printf '%s' "${SRC_FILE##*.}" | tr '[:upper:]' '[:lower:]')

if [[ "$SRC_EXT_LOWER" == "$OUT_EXT_LOWER" && "$SRC_FILE" != *.wav || "$OUT_EXT_LOWER" == "wav" && "$SRC_EXT_LOWER" == "wav" ]]; then
    # Same-family extension: stream-copy (lossless, fast).
    ffmpeg -y -hide_banner -loglevel error -i "$SRC_FILE" -c copy "$FINAL_PATH"
else
    # Format conversion needed (e.g. wav -> mp3, or mp4 -> mp3 audio-only).
    case "$OUT_EXT_LOWER" in
        mp3)
            ffmpeg -y -hide_banner -loglevel error \
                -i "$SRC_FILE" \
                -vn -ac 2 -ar 48000 -b:a 192k \
                -codec:a libmp3lame \
                "$FINAL_PATH"
            ;;
        m4a|aac)
            ffmpeg -y -hide_banner -loglevel error \
                -i "$SRC_FILE" \
                -vn -ac 2 -ar 48000 -b:a 192k \
                -codec:a aac \
                "$FINAL_PATH"
            ;;
        flac)
            ffmpeg -y -hide_banner -loglevel error \
                -i "$SRC_FILE" \
                -vn -ac 2 -ar 48000 -codec:a flac \
                "$FINAL_PATH"
            ;;
        ogg)
            ffmpeg -y -hide_banner -loglevel error \
                -i "$SRC_FILE" \
                -vn -ac 2 -ar 48000 -codec:a libvorbis -q:a 5 \
                "$FINAL_PATH"
            ;;
        wav)
            # wav target: re-encode to 16-bit PCM stereo 48 kHz
            ffmpeg -y -hide_banner -loglevel error \
                -i "$SRC_FILE" \
                -vn -ac 2 -ar 48000 -codec:a pcm_s16le \
                "$FINAL_PATH"
            ;;
        mp4|mkv|mov|webm|m4v)
            if [[ "$SRC_FILE" == *.wav ]]; then
                # Audio-only source: produce a video container with just
                # the audio track (no blank h264 stream).
                ffmpeg -y -hide_banner -loglevel error \
                    -i "$SRC_FILE" \
                    -vn -c:a aac -b:a 192k \
                    "$FINAL_PATH"
            else
                # Real video source: stream-copy video, transcode audio
                # so the new container always gets an aac track.
                ffmpeg -y -hide_banner -loglevel error \
                    -i "$SRC_FILE" \
                    -c:v copy -c:a aac -b:a 192k \
                    "$FINAL_PATH"
            fi
            ;;
        *)
            # Unknown target extension: do a stream copy and let ffmpeg
            # auto-detect the muxer from the suffix.
            ffmpeg -y -hide_banner -loglevel error \
                -i "$SRC_FILE" -c copy "$FINAL_PATH"
            ;;
    esac
fi

# If the user asked for --audio-only but the file we produced has a video
# container extension, FFmpeg will already have stripped the video stream.
DUR=$(ffprobe -hide_banner -v error -show_entries format=duration -of csv=p=0 "$FINAL_PATH" 2>/dev/null || echo "?")
SIZE=$(du -h -- "$FINAL_PATH" 2>/dev/null | cut -f1)

echo ""
echo "[ok] Wrote: $FINAL_PATH"
echo "     Size : ${SIZE:-?}    Duration: ${DUR:-?}s"
# Print the most recent workspace ID (if any) so one-shot users can resume later.
WS_ROOT="${HOME}/.local/share/ai-dubbing/workspaces"
if [[ -d "$WS_ROOT" ]]; then
    LATEST_WS=$(ls -1t "$WS_ROOT" 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_WS" ]]; then
        echo "     Workspace: $LATEST_WS  ($WS_ROOT/$LATEST_WS)"
    fi
fi
