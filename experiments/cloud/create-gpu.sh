#!/usr/bin/env bash
# Retry creating the GPU VM (mestrado-gpu) until a zone has L4 capacity.
#
# L4 GPUs in us-central1 are frequently stocked out. This script tries
# each zone in order and, if all are stocked out, sleeps 30s before the
# next cycle. Exits immediately on non-stockout errors (auth, quota,
# bad flag, etc.) so they surface instead of looping silently.
#
# Usage:
#   bash experiments/cloud/create-gpu.sh
#   bash experiments/cloud/create-gpu.sh us-central1-b,us-central1-c
#
# Ctrl+C to stop. On success the VM name and zone are printed.

set -euo pipefail

ZONES_CSV="${1:-us-central1-a,us-central1-b,us-central1-c,us-central1-f}"
IFS=',' read -ra ZONES <<< "$ZONES_CSV"

INSTANCE="mestrado-gpu"
MACHINE_TYPE="g2-standard-4"
ACCELERATOR="type=nvidia-l4,count=1"
SLEEP=30

attempt=0
while true; do
    for zone in "${ZONES[@]}"; do
        attempt=$((attempt + 1))
        echo "[create-gpu $(date +%H:%M:%S)] attempt $attempt: zone=$zone"

        if err=$(gcloud compute instances create "$INSTANCE" \
            --zone="$zone" \
            --machine-type="$MACHINE_TYPE" \
            --accelerator="$ACCELERATOR" \
            --image-family=ubuntu-2404-lts-amd64 \
            --image-project=ubuntu-os-cloud \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-balanced \
            --maintenance-policy=TERMINATE 2>&1); then
            echo "[create-gpu] created $INSTANCE in $zone"
            exit 0
        fi

        if echo "$err" | grep -qE "ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT"; then
            echo "[create-gpu] $zone: stockout"
            continue
        fi

        echo "[create-gpu] non-stockout error, bailing:"
        echo "$err"
        exit 1
    done

    echo "[create-gpu] all zones stocked out; sleeping ${SLEEP}s"
    sleep "$SLEEP"
done
