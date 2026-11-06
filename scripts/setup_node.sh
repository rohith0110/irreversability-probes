#!/usr/bin/env bash
# One-time setup on a fresh E2E A30 node, from the repo root. Needs HF_TOKEN in the
# environment (with Llama 3.1 access for the llama plan). Then start the queue in tmux:
#   tmux new -s irrev "python -m irrev.run_queue qwen --sync-repo <user>/adnn-results 2>&1 | tee -a logs/queue.out"
set -euo pipefail

nvidia-smi
python3 --version
df -h .

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
# torch 2.15.0 on PyPI is built for CUDA 13; older drivers need a matching wheel index.
DRIVER_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
echo "driver supports CUDA $DRIVER_CUDA"
case "$DRIVER_CUDA" in
  1[3-9].*) TORCH_INDEX="" ;;
  12.[8-9]) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
  12.[6-7]) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
  *) echo "driver CUDA $DRIVER_CUDA is too old for torch 2.15; stop and ask"; exit 1 ;;
esac
if [ -n "$TORCH_INDEX" ]; then pip install torch==2.15.0 --index-url "$TORCH_INDEX"; fi
pip install -r requirements.txt
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU: check the driver against the torch CUDA build"
print(torch.cuda.get_device_name(0), "cuda", torch.version.cuda, "torch", torch.__version__)
EOF

python -m pytest -q tests

PLAN="${1:-qwen}"
MODEL=$([ "$PLAN" = "llama" ] && echo "meta-llama/Llama-3.1-8B-Instruct" || echo "Qwen/Qwen2.5-7B-Instruct")
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', allow_patterns=['*.json', '*.safetensors', 'tokenizer*']))"
echo "setup done for plan $PLAN"
