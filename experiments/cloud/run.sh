#!/usr/bin/env bash
# Launch an experiment study inside a detached tmux session so SSH
# disconnects don't kill the run. Resumes if the target tmux session
# already exists.
#
# Usage: bash experiments/cloud/run.sh <study> [n_seeds] [extra uv-run args...]
#   study    one of: binary, multiclass
#   n_seeds  optional (default 10)
#
# Extra args are forwarded verbatim to the runner, so the predict-cache
# decoupling works here too:
#   Phase A: bash experiments/cloud/run.sh multiclass 10 --dump-predictions /mnt/cache
#   Phase B: bash experiments/cloud/run.sh multiclass 10 --from-predictions /mnt/cache
#
# Results are written under $RESULTS_DIR if set, else experiments/results/.
# Attach with:  tmux attach -t run-<study>

set -euo pipefail

# uv is installed under ~/.local/bin; make sure it's on PATH even when
# this script runs from a non-interactive shell (where ~/.bashrc isn't
# sourced). Harmless if already present.
export PATH="$HOME/.local/bin:$PATH"

STUDY="${1:?study required (binary|multiclass)}"
N_SEEDS="${2:-10}"
shift || true
shift || true

case "$STUDY" in
    binary)     MODULE="experiments.binary.runner" ;;
    multiclass) MODULE="experiments.multiclass.runner" ;;
    *) echo "unknown study: $STUDY" >&2; exit 2 ;;
esac

SESSION="run-$STUDY"
ESCAPED_ARGS=""
for arg in "$@"; do
    ESCAPED_ARGS="$ESCAPED_ARGS $(printf '%q' "$arg")"
done
CMD="export PATH=\"\$HOME/.local/bin:\$PATH\"; uv run python -m $MODULE --n-seeds $N_SEEDS $ESCAPED_ARGS 2>&1 | tee run-$STUDY.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION already exists; attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" "$CMD"
echo "launched $SESSION"
echo "attach:  tmux attach -t $SESSION"
echo "tail:    tail -f run-$STUDY.log"
