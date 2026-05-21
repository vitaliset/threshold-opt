#!/usr/bin/env bash
# Quick status snapshot for an in-flight experiment run.
# Reads the newest run folder under $RESULTS_DIR (default experiments/results/)
# and prints task count, elapsed wall time, throughput, and the tail of the
# tqdm progress bar from run-<study>.log if present.
#
# Run this on whichever machine is executing the experiment (typically the VM).

set -euo pipefail

# Active tmux sessions (experiment runners)
SESSIONS=$(tmux list-sessions 2>/dev/null || true)
if [ -n "$SESSIONS" ]; then
    echo "tmux:    $SESSIONS"
else
    echo "tmux:    no active sessions"
fi
echo

RESULTS_DIR="${RESULTS_DIR:-experiments/results}"

RUN=$(ls -td "$RESULTS_DIR"/run_* 2>/dev/null | head -1 || true)
if [ -z "$RUN" ]; then
    echo "no run folders under $RESULTS_DIR"
    exit 1
fi

DONE=$(find "$RUN" -maxdepth 1 -name '*_seed*.parquet' 2>/dev/null | wc -l)

# Parse start time from the directory name (run_*_YYYYMMDD_HHMMSS) rather
# than stat, which resets every time a file is written inside the dir.
TS=$(basename "$RUN" | grep -oE '[0-9]{8}_[0-9]{6}$')
if [ -n "$TS" ]; then
    START=$(date -d "${TS:0:8} ${TS:9:2}:${TS:11:2}:${TS:13:2}" +%s 2>/dev/null || stat -c %Y "$RUN")
else
    START=$(stat -c %Y "$RUN")
fi
NOW=$(date +%s)
ELAPSED=$((NOW - START))
H=$((ELAPSED/3600)); M=$(((ELAPSED%3600)/60)); S=$((ELAPSED%60))

# Parse study name from run folder. The pipeline names folders
# run_results_<study>_<ts> (or run_<study>_<ts>), so match the known
# study names directly rather than a generic wildcard.
STUDY=$(basename "$RUN" \
    | sed -E 's/^run_(results_)?(binary|multiclass|gap|noise)_[0-9]{8}_[0-9]{6}$/\2/')
LOG="run-$STUDY.log"

# Derive total task count from the log header written by the pipeline
# ("Running N total tasks (D datasets x S seeds)").
TOTAL=""
if [ -f "$LOG" ]; then
    TOTAL=$(grep -m1 'Running [0-9]* total tasks' "$LOG" \
        | grep -oE '[0-9]+ total tasks' | grep -oE '^[0-9]+' || true)
fi

echo "run:     $RUN"
printf "elapsed: %02dh%02dm%02ds\n" "$H" "$M" "$S"
if [ -n "$TOTAL" ]; then
    echo "done:    $DONE / $TOTAL tasks (per-seed parquet files)"
else
    echo "done:    $DONE tasks (per-seed parquet files)"
fi

if [ "$DONE" -gt 0 ] && [ "$ELAPSED" -gt 0 ]; then
    RATE=$(awk -v d="$DONE" -v e="$ELAPSED" 'BEGIN{printf "%.2f", d*60/e}')
    echo "rate:    $RATE tasks/min"
    if [ -n "$TOTAL" ] && [ "$TOTAL" -gt "$DONE" ]; then
        ETA=$(awk -v d="$DONE" -v t="$TOTAL" -v e="$ELAPSED" \
            'BEGIN{r=d/e; rem=t-d; printf "%dh%02dm", int(rem/r/3600), int(rem/r%3600/60)}')
        echo "eta:     ~$ETA remaining"
    fi
    LAST=$(find "$RUN" -maxdepth 1 -name '*_seed*.parquet' -printf '%T@ %p\n' \
        | sort -n | tail -1 | awk '{print $2}')
    echo "last:    $(basename "$LAST") @ $(date -r "$LAST" '+%H:%M:%S')"
fi

if [ -f "$LOG" ]; then
    echo
    echo "--- recent log activity ($LOG) ---"
    tail -c 8192 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -n 5
fi
