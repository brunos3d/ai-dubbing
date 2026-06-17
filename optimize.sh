#!/usr/bin/env bash
# optimize.sh - friendly front-end for the pipeline self-optimization system.
#
# Repeatedly processes the SAME video through the dubbing pipeline (OmniVoice),
# measures how closely the output matches the source (same-language benchmark),
# and searches for the parameter configuration that maximises that quality.
# Runs autonomously, survives failures, resumes where it left off, and reuses a
# single workspace so disk use stays bounded.
#
# Usage:
#   ./optimize.sh run    -m <media> [-l en] [-n N] [options]   # run (default)
#   ./optimize.sh status -m <media> [-l en] [-r name]          # one-shot status
#   ./optimize.sh watch  -m <media> [-l en] [-r name]          # live pipeline log
#   ./optimize.sh best   -m <media> [-l en] [-r name]          # best config so far
#   ./optimize.sh clean  -m <media> [-l en] [-r name]          # delete this run
#   ./optimize.sh help
#
# Common flags:
#   -m, --media PATH        source video/audio                       (required)
#   -l, --language CODE     same-language benchmark code             (default: en)
#   -n, --iterations N      iterations; 0 = infinite until Ctrl-C    (default: 0)
#   -r, --run-name NAME     optimization run id    (default: <slug>-<lang>-<lang>)
#   -s, --seed N            RNG seed                                 (default: 1234)
#       --to-stage NAME     last pipeline stage per iteration  (default: reconstruct)
#       --epsilon F         random-exploration probability          (default: 0.2)
#       --restart-patience N  random restart after N stale steps    (default: 8)
#       --interval SEC      status refresh interval, seconds        (default: 5)
#       --mock              synthetic scorer; no GPU/pipeline (smoke test)
#       --verbose           also stream the raw pipeline log to the terminal
#   --                      everything after is forwarded to optimize_pipeline.py
#
# Examples:
#   ./optimize.sh run -m input/elon-musk-might-be-a-super-villain.mp4 -l en
#   ./optimize.sh run -m input/clip.mp4 -l en -n 50 --interval 3
#   ./optimize.sh status -m input/clip.mp4 -l en
#   ./optimize.sh run -m input/clip.mp4 -l en --mock -n 30   # fast, no GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="$SCRIPT_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
PYSCRIPT="$SCRIPT_DIR/scripts/optimize_pipeline.py"

# --- colors (disabled when not a TTY) --------------------------------------
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
    YEL=$'\033[33m'; BLU=$'\033[34m'; CYN=$'\033[36m'; RST=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GRN=""; YEL=""; BLU=""; CYN=""; RST=""
fi

strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }

print_usage() { sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- flag parsing -----------------------------------------------------------
CMD="run"
case "${1:-}" in
    run|status|watch|best|clean|help|-h|--help) CMD="$1"; shift || true ;;
esac
[ "$CMD" = "-h" ] || [ "$CMD" = "--help" ] && CMD="help"

MEDIA=""; LANG_CODE="en"; ITERS="0"; RUN_NAME=""; SEED="1234"
TO_STAGE="reconstruct"; EPSILON="0.2"; RESTART="8"; INTERVAL="5"
MOCK=0; VERBOSE=0; PASSTHRU=()

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--media)          MEDIA="$2"; shift 2 ;;
        -l|--language)       LANG_CODE="$2"; shift 2 ;;
        -n|--iterations)     ITERS="$2"; shift 2 ;;
        -r|--run-name)       RUN_NAME="$2"; shift 2 ;;
        -s|--seed)           SEED="$2"; shift 2 ;;
        --to-stage)          TO_STAGE="$2"; shift 2 ;;
        --epsilon)           EPSILON="$2"; shift 2 ;;
        --restart-patience)  RESTART="$2"; shift 2 ;;
        --interval)          INTERVAL="$2"; shift 2 ;;
        --mock)              MOCK=1; shift ;;
        --verbose)           VERBOSE=1; shift ;;
        --)                  shift; PASSTHRU=("$@"); break ;;
        *) echo "${RED}Unknown flag: $1${RST}" >&2; exit 2 ;;
    esac
done

if [ "$CMD" = "help" ]; then print_usage; exit 0; fi

if [ -z "$MEDIA" ]; then
    echo "${RED}error:${RST} --media is required (see './optimize.sh help')" >&2
    exit 2
fi

# Build the common args that identify the run (resolve_run_dir uses these).
COMMON_ARGS=(--media "$MEDIA" --language "$LANG_CODE")
[ -n "$RUN_NAME" ] && COMMON_ARGS+=(--run-name "$RUN_NAME")

# Resolve the run directory once (needs the heavier import, done a single time).
RUN_DIR="$("$PY" - "$MEDIA" "$LANG_CODE" "$RUN_NAME" <<'PYEOF'
import sys
from src.optimization.optimizer import OptimizerConfig
media, lang, name = sys.argv[1], sys.argv[2], (sys.argv[3] or None)
print(OptimizerConfig(input_path=media, language=lang, run_name=name).resolved_run_dir())
PYEOF
)"
LOG_FILE="$RUN_DIR/run.log"
HIST="$RUN_DIR/history.jsonl"

# --- pure-stdlib formatter: print blocks for new history records ------------
# args: RUN_DIR  START_LINE   (prints records with line-index >= START_LINE)
print_new_records() {
    "$PY" - "$RUN_DIR" "$1" "$BOLD" "$DIM" "$GRN" "$YEL" "$RED" "$CYN" "$RST" <<'PYEOF'
import json, os, sys
run_dir, start = sys.argv[1], int(sys.argv[2])
BOLD, DIM, GRN, YEL, RED, CYN, RST = sys.argv[3:10]
hist = os.path.join(run_dir, "history.jsonl")
best_path = os.path.join(run_dir, "best.json")
best = None
if os.path.exists(best_path):
    try: best = json.load(open(best_path))
    except Exception: best = None
lines = []
if os.path.exists(hist):
    lines = [l for l in open(hist).read().splitlines() if l.strip()]
for rec_line in lines[start:]:
    try: r = json.loads(rec_line)
    except Exception: continue
    it = r.get("iteration"); src = r.get("source", "?")
    score = r.get("score"); status = r.get("status", "?")
    took = r.get("duration_s"); fs = r.get("from_stage")
    bscore = best.get("score") if best else None
    bit = best.get("iteration") if best else None
    is_best = best and best.get("iteration") == it
    tag = f"{GRN}NEW BEST{RST}" if is_best else (f"{RED}FAILED{RST}" if status=="failed" else f"{DIM}ok{RST}")
    sc = f"{score:.5f}" if isinstance(score,(int,float)) else "  n/a"
    bs = f"{bscore:.5f}" if isinstance(bscore,(int,float)) else "n/a"
    print(f"{BOLD}── iter {it:<4}{RST} [{CYN}{src}{RST}]  score {YEL}{sc}{RST}  {tag}   best {bs} (iter {bit})")
    extra = []
    if fs: extra.append(f"from:{fs}")
    if isinstance(took,(int,float)): extra.append(f"{took:.1f}s")
    if r.get("error"): extra.append(f"{RED}{r['error'][:60]}{RST}")
    if extra: print("   " + "   ".join(extra))
    m = r.get("metrics") or {}
    parts = []
    for k in ("timing_accuracy","slot_fit","speaker_similarity","speaker_consistency",
              "ending_quality","continuity","prosodic_similarity","reconstruction_quality"):
        v = m.get(k)
        if isinstance(v,(int,float)):
            short = {"timing_accuracy":"timing","slot_fit":"slot","speaker_similarity":"spkSim",
                     "speaker_consistency":"spkCons","ending_quality":"end","continuity":"cont",
                     "prosodic_similarity":"pros","reconstruction_quality":"recon"}[k]
            parts.append(f"{short} {v:.2f}")
    if parts: print("   " + DIM + "  ".join(parts) + RST)
# emit the new line count on the very last stdout line, prefixed for bash to read
print(f"__COUNT__ {len(lines)}")
PYEOF
}

current_activity() {
    # Last meaningful pipeline line (stage banner or per-segment progress).
    [ -f "$LOG_FILE" ] || return 0
    tail -n 60 "$LOG_FILE" | strip_ansi \
        | grep -aoE '(\[[0-9]+/[0-9]+\].*|Segment +[0-9]+:.*|Loading OmniVoice.*|Reclustered speakers.*|Aligning .*|Reconstructed .*)' \
        | tail -n 1 || true
}

cmd_status() {
    "$PY" "$PYSCRIPT" "${COMMON_ARGS[@]}" --report
}

cmd_best() {
    if [ -f "$RUN_DIR/best.json" ]; then
        echo "${BOLD}Best configuration so far${RST} ($RUN_DIR/best.json):"
        "$PY" -m json.tool "$RUN_DIR/best.json"
    else
        echo "${YEL}No best.json yet — run some iterations first.${RST}"
    fi
}

cmd_watch() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "${YEL}No live log at $LOG_FILE (is a run active?).${RST}"; exit 0
    fi
    echo "${DIM}Tailing $LOG_FILE  (Ctrl-C to stop)${RST}"
    tail -n 40 -f "$LOG_FILE"
}

cmd_clean() {
    if [ -d "$RUN_DIR" ]; then
        echo "${YEL}Deleting run dir:${RST} $RUN_DIR"
        rm -rf "$RUN_DIR"
        echo "${GRN}done${RST}  (workspace artifacts are kept; only optimization history removed)"
    else
        echo "Nothing to clean at $RUN_DIR"
    fi
}

cmd_run() {
    mkdir -p "$RUN_DIR"
    local extra=()
    [ "$MOCK" = "1" ] && extra+=(--mock)
    [ ${#PASSTHRU[@]} -gt 0 ] && extra+=("${PASSTHRU[@]}")

    echo "${BOLD}${BLU}AI-Dubbing pipeline self-optimization${RST}"
    echo "  media     : $MEDIA"
    echo "  benchmark : ${LANG_CODE} → ${LANG_CODE} (same-language)"
    echo "  iterations: $([ "$ITERS" -le 0 ] 2>/dev/null && echo 'infinite (Ctrl-C to stop)' || echo "$ITERS")"
    echo "  run dir   : $RUN_DIR"
    echo "  live log  : $LOG_FILE"
    echo "  status    : ./optimize.sh status -m '$MEDIA' -l $LANG_CODE"
    echo "${DIM}────────────────────────────────────────────────────────────${RST}"

    # Launch the optimizer; full pipeline output streams to the log file.
    : > "$LOG_FILE"
    "$PY" "$PYSCRIPT" "${COMMON_ARGS[@]}" \
        --iterations "$ITERS" --seed "$SEED" --to-stage "$TO_STAGE" \
        --epsilon "$EPSILON" --restart-patience "$RESTART" \
        "${extra[@]}" >> "$LOG_FILE" 2>&1 &
    local pypid=$!

    cleanup() {
        # Remove trap immediately to prevent re-entry on multiple CTRL+C
        trap - INT TERM
        
        if kill -0 "$pypid" 2>/dev/null; then
            echo; echo "${YEL}Stopping… (run is resumable: re-run the same command)${RST}"
            kill -INT "$pypid" 2>/dev/null || true
            
            # Wait up to 5 seconds for graceful shutdown
            local waited=0
            while kill -0 "$pypid" 2>/dev/null && [ $waited -lt 5 ]; do
                sleep 1
                waited=$((waited + 1))
            done
            
            # Force kill if still running
            if kill -0 "$pypid" 2>/dev/null; then
                echo "${YEL}Force stopping…${RST}"
                kill -KILL "$pypid" 2>/dev/null || true
                wait "$pypid" 2>/dev/null || true
            fi
        fi
        
        # Clean up tail process if running
        [ -n "${TAILPID:-}" ] && kill "$TAILPID" 2>/dev/null || true
        
        exit 0
    }
    trap cleanup INT TERM

    [ "$VERBOSE" = "1" ] && { tail -n +1 -f "$LOG_FILE" & TAILPID=$!; }

    local last_count=0 last_activity=""
    while kill -0 "$pypid" 2>/dev/null; do
        if [ -f "$HIST" ]; then
            local out count
            out="$(print_new_records "$last_count")"
            count="$(printf '%s\n' "$out" | sed -n 's/^__COUNT__ //p' | tail -n1)"
            printf '%s\n' "$out" | grep -v '^__COUNT__' || true
            [ -n "${count:-}" ] && last_count="$count"
        fi
        if [ "$VERBOSE" != "1" ]; then
            local act; act="$(current_activity)"
            if [ -n "$act" ] && [ "$act" != "$last_activity" ]; then
                printf '   %s⏳ %s%s\n' "$DIM" "$act" "$RST"
                last_activity="$act"
            fi
        fi
        sleep "$INTERVAL"
    done
    wait "$pypid" 2>/dev/null || true
    [ -n "${TAILPID:-}" ] && kill "$TAILPID" 2>/dev/null || true

    # Flush any final records that landed between the last poll and exit.
    if [ -f "$HIST" ]; then
        print_new_records "$last_count" | grep -v '^__COUNT__' || true
    fi

    echo "${DIM}────────────────────────────────────────────────────────────${RST}"
    echo "${BOLD}Final status:${RST}"
    cmd_status
}

case "$CMD" in
    run)       cmd_run ;;
    status)    cmd_status ;;
    watch)     cmd_watch ;;
    best)      cmd_best ;;
    promote)   cmd_promote "$@" ;;
    show-best) cmd_show_best "$@" ;;
    clean)     cmd_clean ;;
    *)         print_usage ;;
esac
s%s\n' "$DIM" "$act" "$RST"
                last_activity="$act"
            fi
        fi
        sleep "$INTERVAL"
    done
    wait "$pypid" 2>/dev/null || true
    [ -n "${TAILPID:-}" ] && kill "$TAILPID" 2>/dev/null || true

    # Flush any final records that landed between the last poll and exit.
    if [ -f "$HIST" ]; then
        print_new_records "$last_count" | grep -v '^__COUNT__' || true
    fi

    echo "${DIM}────────────────────────────────────────────────────────────${RST}"
    echo "${BOLD}Final status:${RST}"
    cmd_status
}

case "$CMD" in
    run)    cmd_run ;;
    status) cmd_status ;;
    watch)  cmd_watch ;;
    best)   cmd_best ;;
    clean)  cmd_clean ;;
    *)      print_usage ;;
esac
