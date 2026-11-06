#!/usr/bin/env bash
# Laptop side: pack code and data into two tars for scp to an E2E node (no git on the node).
# Usage (Git Bash, repo root): bash scripts/pack_upload.sh   -> upload/code.tar, upload/data.tar
set -euo pipefail
mkdir -p upload
tar --exclude='__pycache__' -cf upload/code.tar irrev tests scripts configs docs requirements.txt README.md
tar -cf upload/data.tar data/pairs.jsonl data/smoke_pairs.jsonl data/rq5/transitions.jsonl \
    data/external/bytesized32/test/test.jsonl data/external/bytesized32/src/data/games \
    data/external/bytesized32/src/data/playthroughs
ls -la upload
echo "next: scp upload/code.tar upload/data.tar <ssh-alias>:/root/adnn/   (see docs/node_runbook.md)"
