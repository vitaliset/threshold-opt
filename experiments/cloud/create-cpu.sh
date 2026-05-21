#!/usr/bin/env bash
# Retry creating the CPU VM (mestrado-runner) until a zone has n2-highcpu-80 capacity.
#
# us-central1-a is frequently stocked out on n2-highcpu-80. This script tries
# each zone in order and, if all are stocked out, sleeps 30s before the next
# cycle. Exits immediately on non-stockout errors (auth, quota, bad flag,
# etc.) so they surface instead of looping silently.
#
# Usage:
#   bash experiments/cloud/create-cpu.sh
#   bash experiments/cloud/create-cpu.sh us-central1-b,us-central1-c
#
# Ctrl+C to stop. On success the VM name and zone are printed.

set -euo pipefail

ZONES_CSV="${1:-us-central1-a,us-central1-b,us-central1-c,us-central1-f}"
IFS=',' read -ra ZONES <<< "$ZONES_CSV"

INSTANCE="mestrado-runner"
MACHINE_TYPE="n2-highcpu-80"
SLEEP=30

attempt=0
while true; do
    for zone in "${ZONES[@]}"; do
        attempt=$((attempt + 1))
        echo "[create-cpu $(date +%H:%M:%S)] attempt $attempt: zone=$zone"

        if err=$(gcloud compute instances create "$INSTANCE" \
            --zone="$zone" \
            --machine-type="$MACHINE_TYPE" \
            --image-family=ubuntu-2404-lts-amd64 \
            --image-project=ubuntu-os-cloud \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-balanced 2>&1); then
            echo "[create-cpu] created $INSTANCE in $zone"
            exit 0
        fi

        if echo "$err" | grep -qE "ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT"; then
            echo "[create-cpu] $zone: stockout"
            continue
        fi

        echo "[create-cpu] non-stockout error, bailing:"
        echo "$err"
        exit 1
    done

    echo "[create-cpu] all zones stocked out; sleeping ${SLEEP}s"
    sleep "$SLEEP"
done
