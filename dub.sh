#!/usr/bin/env bash
# dub.sh - convenience wrapper for the multilingual dubbing pipeline
#
# By default this produces the FULL DUBBED VIDEO in output/final_video.mp4
# (when the source has a video stream) — exactly what most users want.
# Use a flag if you'd rather take the audio-only track.
#
# Usage:
#   ./dub.sh <input> <source-language> <target-language> [options]
#
# Examples:
#   ./dub.sh input/video.mp4 pt en                       # -> output/final_video.mp4 (+ final_audio.wav)
#   ./dub.sh input/video.mp4 en pt-BR --audio-only      # -> output/final_audio.wav only
#   ./dub.sh input/song.mp3 en es --emit=audio          # -> output/final_audio.wav
#   ./dub.sh input/video.mp4 en es -- --whisper-model=medium --target-lufs=-14
#
# Anything after the literal `--` is forwarded to main.py, so the usual
# flags are available: --audio-only, --emit, --whisper-model,
# --target-lufs, --hf-token, --workdir, --output-dir, etc.
# Run `./dub.sh --help` to see them via the underlying CLI.

set -euo pipefail

print_usage() {
    cat <<'EOF'
dub.sh - local-first multilingual dubbing pipeline

  ./dub.sh <input> <source-language> <target-language> [options]

Default output: the full dubbed VIDEO in <output-dir>/final_video.mp4
(when the source has a video stream). The dubbed audio is always
written to <output-dir>/final_audio.wav.

Options (place these AFTER the three positional args):
  --audio-only        Skip the video remux; emit only final_audio.wav.
  --emit <mode>       One of: auto (default), audio, video, both.
                      'auto' / 'both' / 'video' deliver the full video;
                      'audio' delivers only the audio.

  -- <main.py flags>  Forward arbitrary flags to main.py
                      e.g. -- --whisper-model=medium --target-lufs=-14

Examples:
  ./dub.sh input/video.mp4 pt en
  ./dub.sh input/video.mp4 en pt-BR --audio-only
  ./dub.sh input/episode.mkv en es -- --whisper-model=medium
EOF
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    print_usage
    exit 0
fi

INPUT="$1"
SRC="$2"
TGT="$3"
shift 3 || true

# Collect any -- separator and everything after it
PYTHON_ARGS=()
USER_ARGS=()
SEEN_SEP=0
for arg in "$@"; do
    if [[ $SEEN_SEP -eq 0 && "$arg" == "--" ]]; then
        SEEN_SEP=1
        continue
    fi
    if [[ $SEEN_SEP -eq 1 ]]; then
        USER_ARGS+=("$arg")
    else
        PYTHON_ARGS+=("$arg")
    fi
done

# Translate friendly flags into main.py flags.
EMIT=""
EMIT_NEXT=0
if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    for arg in "${PYTHON_ARGS[@]}"; do
        case "$arg" in
            --audio-only) USER_ARGS+=("--audio-only") ;;
            --emit=*)     EMIT="${arg#--emit=}" ;;
            --emit)       EMIT_NEXT=1 ;;
            *)
                if [[ $EMIT_NEXT -eq 1 ]]; then
                    EMIT="$arg"
                    EMIT_NEXT=0
                else
                    USER_ARGS+=("$arg")
                fi
                ;;
        esac
    done
fi

if [[ -n "$EMIT" ]]; then
    USER_ARGS+=("--emit" "$EMIT")
fi

cd "$(dirname "$0")"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

echo ">> Dubbing: $INPUT ($SRC -> $TGT)"
if [[ -t 1 ]]; then
    echo ">> Output  : $(pwd)/output/"
fi

# Default: deliver the full video (and audio as a byproduct).
# The CLI's --emit=auto / --emit=video handles this; --audio-only overrides it.
exec "$PYTHON_BIN" main.py \
    --input "$INPUT" \
    --source-language "$SRC" \
    --target-language "$TGT" \
    "${USER_ARGS[@]}"
