# Do AI agents know which actions cannot be undone?

ADNN project, option 1. We measure whether open-weight LLM agents (Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct) represent the reversibility of actions (RQ1, linear probes), say it
when asked (RQ2), act on it (RQ3, 60 paired tasks), and whether steering along the probe
direction changes behaviour (RQ4). The analysis plan was fixed before any result:
[docs/analysis_plan.md](docs/analysis_plan.md).

## Layout

| Path | What it is |
| --- | --- |
| `irrev/sandbox.py` | Simulated file system, recycle bin, archive, backups, outbox and wallet with `save()`/`restore()`; reversibility design rules |
| `irrev/sandbox_data.py` | Sandbox (state, action) pairs; minimal groups share one action across three states |
| `irrev/sw_label.py` | ScienceWorld labelling by reset-and-replay BFS to depth 3 against a null baseline |
| `irrev/dataset.py` | Merges both testbeds into `data/pairs.jsonl`, writes the human agreement sheet |
| `irrev/tasks.py` | The 60 paired behaviour tasks |
| `irrev/hidden.py` | Per-layer hidden states (last action token and mean over action tokens) |
| `irrev/probes.py` | Probes, all controls and ablations, layer selection, steering direction, RQ2 comparison |
| `irrev/stated.py` | RQ2 stated answers for every pair and every task version |
| `irrev/agent.py` | Batched lockstep agent loop, steering hook, run summaries |
| `irrev/behaviour.py` | RQ3/RQ4 metrics, blindness gap, McNemar tests |
| `irrev/rq5.py` | RQ5: ByteSized32 transition labelling, prediction NLL, AUROC analysis |
| `irrev/figures.py` | Every figure and table in the report |
| `irrev/run_queue.py` | Unattended per-node job queue (heartbeat, hourly sync) |
| `modal_app.py` | Modal smoke test (development only) |
| `tests/` | Sandbox label verification, task pairs, agent bookkeeping, prompts |

## Setup

Python 3.11, Java 1.8+ (ScienceWorld), exact versions in `requirements.txt`.

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                    # CPU-only laptop: install torch==2.15.0 from the PyTorch CPU index first
python -m pytest -q tests
```

Llama 3.1/3.2 are gated: accept the licence on Hugging Face and set `HF_TOKEN`.

## Reproduce

1. Data (CPU):
   ```bash
   python -m irrev.sandbox_data                       # data/raw/sandbox_pairs.jsonl
   python -m irrev.sw_label --workers 8               # data/raw/scienceworld_pairs.jsonl (hours)
   python -m irrev.sw_label --thermal --workers 8 --out data/raw/scienceworld_thermal.jsonl
   python -m irrev.dataset                            # data/pairs.jsonl, data/agreement/
   ```
2. GPU jobs, one A30 per model (`scripts/setup_node.sh`, then inside tmux):
   ```bash
   python -m irrev.run_queue qwen  --sync-repo <user>/adnn-results
   python -m irrev.run_queue llama --sync-repo <user>/adnn-results
   ```
3. Analysis (CPU), per model:
   ```bash
   python -m irrev.probes all --hidden results/qwen7b/hidden --stated results/qwen7b/stated.jsonl --out results/qwen7b/probes
   python -m irrev.behaviour --runs results/qwen7b/agent --stated results/qwen7b/stated.jsonl --out results/qwen7b/behaviour.json
   python -m irrev.behaviour --runs results/qwen7b/steering --stated results/qwen7b/stated.jsonl --out results/qwen7b/steering_behaviour.json
   python -m irrev.rq5 analyze --nll results/qwen7b/rq5/nll.jsonl --out results/qwen7b/rq5/report.json
   python -m irrev.figures
   ```

RQ5 data (not redistributed, no licence published): download `data/test.zip`,
`data/games/` and `data/playthroughs/` from github.com/cognitiveailab/GPT-simulator into
`data/external/bytesized32/`, then `py -3.12 -m irrev.rq5 label` (some game files need
Python 3.12+). The node runbook is in [docs/node_runbook.md](docs/node_runbook.md).

All decoding is greedy, seeds are fixed (0), and every prompt and raw model output is
saved next to its results together with GPU, driver, CUDA, torch and transformers versions.

## Safety

Every irreversible action happens inside the simulated sandbox. Nothing connects to real
email, files outside the sandbox, accounts or paid APIs.
