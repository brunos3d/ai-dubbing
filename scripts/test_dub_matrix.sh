#!/usr/bin/env bash
# Run a battery of dub.sh scenarios and report results.
set -u

DUB="./dub.sh"
SRC_FILE="/tmp/dub-test/bruno.mp4"

rm -rf /tmp/dub-test /tmp/dub-out 2>/dev/null
mkdir -p /tmp/dub-test /tmp/dub-out
cp input/video.mp4 "$SRC_FILE"

run_case() {
    local label="$1"
    shift
    local args=("$@")
    echo
    echo "================================================================"
    echo "  $label"
    echo "  cmd: $DUB ${args[*]}"
    echo "================================================================"
    "$DUB" "${args[@]}" 2>&1 | tail -3
}

run_case "1. default next to input"   "$SRC_FILE" pt en
run_case "2. explicit output path"    "$SRC_FILE" pt en /tmp/dub-out/result.mp4
run_case "3. .wav extension → audio"  "$SRC_FILE" pt en /tmp/dub-test/audio.wav
run_case "4. .mp3 format convert"     "$SRC_FILE" pt en /tmp/dub-test/audio.mp3
run_case "5. .flac format convert"    "$SRC_FILE" pt en /tmp/dub-test/audio.flac
run_case "6. .mkv container remux"    "$SRC_FILE" pt en /tmp/dub-test/video.mkv
run_case "7. --audio-only + .mp4"     "$SRC_FILE" pt en /tmp/dub-test/just-audio.mp4 --audio-only
run_case "8. different target lang"   "$SRC_FILE" pt fr
run_case "9. collision: re-run same"  "$SRC_FILE" pt en
run_case "10. collision: re-run again" "$SRC_FILE" pt en
run_case "11. --emit=video override"  "$SRC_FILE" pt en /tmp/dub-test/want-video.mp3 --emit=video
run_case "12. --emit=audio + .mp4"    "$SRC_FILE" pt en /tmp/dub-test/want-audio.mp4 --emit=audio

echo
echo "================================================================"
echo "  Final test directory state"
echo "================================================================"
ls -la /tmp/dub-test/ 2>&1
ls -la /tmp/dub-out/ 2>&1
