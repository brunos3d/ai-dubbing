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
dub.sh - ffmpeg-style multilingual dubbing pipeline

  dub.sh <input> <source-language> <target-language> [output-file] [options]

Examples:
  dub.sh video.mp4 pt en                       # -> ./video-dub-en.mp4
  dub.sh /data/in.mp4 pt en ~/dubbed.mp4      # -> ~/dubbed.mp4
  dub.sh podcast.mp3 en es out.mp3            # -> ./out.mp3
  dub.sh video.mp4 pt en -- --whisper-model=medium

Output path:
  4th positional arg   explicit output file (extension drives audio/video)
  (omitted)             <input-stem>-dub-<tgt>.<input-ext> next to the input
  existing target       auto-incremented with -(1), -(2), ...

Format:
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
# we forward to main.py.
EXPLICIT_OUTPUT=""
MAIN_FLAGS=()
for arg in "${PRE_DASH[@]+"${PRE_DASH[@]}"}"; do
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

# Suppress the CLI's own success banner — dub.sh will print its own.
DUB_SHOW_BANNER=1 "$PYTHON_BIN" main.py \
    --input "$INPUT" \
    --source-language "$SRC" \
    --target-language "$TGT" \
    "${PYTHON_FLAGS[@]}"

# ------------------------------------------------ deliver to final path --

# Decide which artefact the pipeline produced.
SRC_FILE=""
# 1. Prefer the format implied by the user's flag / effective emit.
case "$EFFECTIVE_EMIT" in
    audio)
        [[ -f "output/final_audio.wav" ]] && SRC_FILE="output/final_audio.wav"
        ;;
    *)
        if [[ -f "output/final_video.mp4" ]]; then
            SRC_FILE="output/final_video.mp4"
        elif [[ -f "output/final_audio.wav" ]]; then
            SRC_FILE="output/final_audio.wav"
        fi
        ;;
esac
# 2. Fall back: take whatever exists.
if [[ -z "$SRC_FILE" ]]; then
    for cand in output/final_video.mp4 output/final_audio.wav; do
        if [[ -f "$cand" ]]; then SRC_FILE="$cand"; break; fi
    done
fi
if [[ -z "$SRC_FILE" ]]; then
    echo "ERROR: pipeline produced no output file in output/" >&2
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
