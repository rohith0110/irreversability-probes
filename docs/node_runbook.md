# E2E A30 runbook (one node runs both models)

Measured on Modal A10 with Qwen2.5-7B: hidden states 618 pairs in 1.5 min, RQ2 200 pairs
in 1.1 min, 24 agent episodes in 1 min, RQ5 64 transitions in 1.5 min. Expected on one
A30: Qwen plan about 1.5-2 h (steering sweep is the longest part), Llama plan about 1 h.
The node can be deleted as soon as both plans are done and pulled.

## Human steps

1. Laptop, repo root (Git Bash): `bash scripts/pack_upload.sh` -> `upload/code.tar`, `upload/data.tar`.
2. Create the node, add the SSH key, add an SSH alias (for example `adnn-a30`).
3. `ssh adnn-a30 "mkdir -p /root/adnn"` then
   `scp upload/code.tar upload/data.tar adnn-a30:/root/adnn/`
4. On the node:
   ```bash
   cd /root/adnn && tar -xf code.tar && tar -xf data.tar
   export HF_TOKEN=...            # token with Llama 3.1 access; also used for the hourly sync
   bash scripts/setup_node.sh llama   # installs the torch build for the node's driver, runs tests, downloads models
   ```
   `setup_node.sh` downloads one model; download the other with
   `python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"`.
5. Start both plans one after the other in one tmux session (never two models at once):
   ```bash
   tmux new -s irrev "source .venv/bin/activate && export HF_TOKEN=$HF_TOKEN && \
     python -m irrev.run_queue qwen --sync-repo rohith1221/adnn-irrev-results && \
     python -m irrev.run_queue llama --sync-repo rohith1221/adnn-irrev-results; bash"
   ```
6. Check progress any time: `ssh adnn-a30 cat /root/adnn/logs/qwen/heartbeat.json`
   (or `logs/llama/...`), job outcomes in `logs/<plan>/status.json`.
7. Laptop: `bash scripts/pull_results.sh adnn-a30 --light` for quick checks; at the end
   `bash scripts/pull_results.sh adnn-a30` (includes hidden states). Results are also
   synced hourly to the private HF dataset repo.
8. Check the pulled files on the laptop (`status.json` all `exit 0`), then delete the node.

## If something fails

Every job resumes from its own checkpoint. Rerun only the failed ones, for example
`python -m irrev.run_queue qwen --only steering`. A failed job never blocks later jobs.
