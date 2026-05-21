#!/usr/bin/env bash
# Pull experiment results from a GCP VM to the local machine via
# `gcloud compute scp` (GCP's SSH tunnels through IAP, so plain rsync
# over ssh is awkward).
#
# Run this on your LAPTOP, not the VM.
#
# Usage: bash experiments/cloud/pull-results.sh [instance] [zone] [remote_user]
#   instance     defaults to mestrado-runner
#   zone         defaults to us-central1-a
#   remote_user  defaults to the current local username
#
# Copies into ./experiments/results/ on the local side.
#
# Note: on Windows, `gcloud compute scp --recurse` shells out to PuTTY's
# pscp, which (a) doesn't expand `~`, and (b) refuses `.`/`..` directory
# entries in recursive mode ("security violation" error). We sidestep
# both by tarring on the remote side and streaming the archive through
# `gcloud compute ssh` into a local `tar x`.

set -euo pipefail

INSTANCE="${1:-mestrado-runner}"
ZONE="${2:-us-central1-a}"
REMOTE_USER="${3:-${USER:-$(whoami)}}"
REMOTE_PARENT="/home/$REMOTE_USER/mestrado-draft/experiments"
LOCAL_DIR="experiments"

mkdir -p "$LOCAL_DIR/results"

gcloud compute ssh "$REMOTE_USER@$INSTANCE" --zone="$ZONE" \
    --command="tar czf - -C $REMOTE_PARENT results" \
    | tar xzf - -C "$LOCAL_DIR"
