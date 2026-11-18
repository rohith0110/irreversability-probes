# Do AI agents know which actions cannot be undone?

ADNN project, option 1. We measure whether open-weight LLM agents (Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, and Qwen2.5-14B-Instruct for model size) represent the reversibility
of actions (RQ1, linear probes), say it when asked (RQ2), act on it (RQ3, 60 paired tasks
plus 54 ToolEmu cases), whether steering along the probe direction changes behaviour (RQ4),
and whether next-state uncertainty tracks reversibility (RQ5). The analysis plan was fixed
before any result: [docs/analysis_plan.md](docs/analysis_plan.md) (its change log lists
everything added later). Dataset: [docs/data_card.md](docs/data_card.md).

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
| `irrev/toolemu.py` | 54 ToolEmu curated cases rebuilt as paired sandbox tasks; external probe test |
| `irrev/demo.py` | Recorded demo: replays a saved agent run on both versions of a task |
| `irrev/figures.py` | Every figure and table in the report |
| `irrev/run_queue.py` | Unattended per-node job queue (heartbeat, hourly sync) |
| `modal_app.py` | Modal runner: the reported GPU runs (L40S) and the development smoke tests |
| `tests/` | Sandbox label verification, task pairs, ToolEmu pairs, agent bookkeeping, prompts |
| `report/` | LaTeX source of the paper (ACL style) |
| `results/` | Saved probe scores, agent logs, stated answers, RQ5 scores (hidden-state arrays are not committed) |
| `figures/` | Every figure and table, made by `irrev.figures` |

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
   python -m irrev.toolemu pairs                      # data/toolemu_pairs.jsonl (needs ToolEmu all_cases.json, below)
   ```
2. GPU jobs, one queue per model. The reported runs used Modal L40S GPUs:
   ```bash
   modal secret create adnn-hf HF_TOKEN=<token>       # Llama is gated
   modal run --detach modal_app.py --queues qwen,llama,qwen14b
   modal volume get adnn-results final/results/qwen7b results/   # likewise llama8b, qwen14b
   ```
   The same queue runs on any CUDA machine: `python -m irrev.run_queue qwen` (see
   [docs/node_runbook.md](docs/node_runbook.md)).
3. Analysis (CPU), per model (`qwen7b`, `llama8b`, `qwen14b`):
   ```bash
   M=qwen7b
   python -m irrev.probes all --hidden results/$M/hidden --stated results/$M/stated.jsonl --out results/$M/probes
   python -m irrev.probes ci  --hidden results/$M/hidden --out results/$M/probes
   python -m irrev.behaviour --runs results/$M/agent --stated results/$M/stated.jsonl --out results/$M/behaviour.json
   python -m irrev.behaviour --runs results/$M/steering --stated results/$M/stated.jsonl --out results/$M/steering_behaviour.json
   python -m irrev.behaviour --runs results/$M/toolemu_agent --stated results/$M/toolemu_stated.jsonl --out results/$M/toolemu_behaviour.json
   python -m irrev.toolemu analyze --model $M
   python -m irrev.rq5 analyze --nll results/$M/rq5/nll.jsonl --out results/$M/rq5/report.json
   python -m irrev.figures                            # figures/ (every figure and table)
   python -m irrev.agreement                          # once both annotators have labelled data/agreement/
   ```
4. Report and demo:
   ```bash
   cd report && latexmk -pdf main.tex
   python -m irrev.demo --model qwen7b --task free_space_00 --delay 1
   ```

ToolEmu cases (Apache-2.0): download `assets/all_cases.json` from github.com/ryoungj/ToolEmu
into `data/external/toolemu/`. The emulator is never run.

RQ5 data (not redistributed, no licence published): download `data/test.zip`,
`data/games/` and `data/playthroughs/` from github.com/cognitiveailab/GPT-simulator into
`data/external/bytesized32/`, then `py -3.12 -m irrev.rq5 label` (some game files need
Python 3.12+). The node runbook is in [docs/node_runbook.md](docs/node_runbook.md).

All decoding is greedy, seeds are fixed (0), and every prompt and raw model output is
saved next to its results together with GPU, driver, CUDA, torch and transformers versions.

## Safety

Every irreversible action happens inside the simulated sandbox. Nothing connects to real
email, files outside the sandbox, accounts or paid APIs.
