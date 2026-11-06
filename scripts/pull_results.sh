#!/usr/bin/env bash
# Laptop side: stream results/ and logs/ back from the node over SSH (no files are deleted on
# either side). Usage: bash scripts/pull_results.sh <ssh-alias> [--light]
# --light skips the large hidden-state arrays for quick hourly checks.
set -euo pipefail
HOST="$1"
EXCLUDE=""
if [ "${2:-}" = "--light" ]; then EXCLUDE="--exclude=*.npy"; fi
ssh "$HOST" "cd /root/adnn && tar $EXCLUDE -cf - results logs" | tar -xf - -C .
echo "pulled into $(pwd)/results and $(pwd)/logs"
cat logs/*/status.json 2>/dev/null || true
