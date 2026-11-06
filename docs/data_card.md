# Data card: reversibility-labelled (state, action) pairs

File: `data/pairs.jsonl` (JSON lines). One record per (state, action) pair.

| Field | Meaning |
| --- | --- |
| `id` | unique id (`sb-...` sandbox, `sw-...` ScienceWorld) |
| `environment` | `sandbox` or `scienceworld` |
| `state_text` | the state as the model sees it |
| `action_text` | the candidate action (sandbox: one JSON tool call; ScienceWorld: one command) |
| `label` | `reversible`, `partially_reversible` or `irreversible` at horizon 3 |
| `horizon` | horizon of `label` (3) |
| `source` | `sandbox_design_rule` or `scienceworld_bfs_replay` |
| `scenario_id` | split unit; all pairs of one scenario share a split |
| `pair_group` | minimal-pair group, or null |
| `verb` | tool name or ScienceWorld verb (used by the control task) |
| `task_text` | sandbox task description (empty for ScienceWorld, where it is inside `state_text`) |
| `labels_by_horizon` | labels at horizons 1, 2, 3 |

## How it was built

- **Sandbox** (`irrev/sandbox_data.py`): random states (files, settings, backups, wallet).
  Minimal groups put one action JSON in three states that differ only in the setting that
  decides its reversibility (recycle bin, backup, version history, email undo/recall,
  payment cancellation/refund). Labels come from design rules verified by an explicit
  restore search (`tests/test_sandbox.py`).
- **ScienceWorld** (`irrev/sw_label.py`): states sampled from gold-path prefixes of all 30
  tasks (5 variations each), actions sampled per verb plus the gold next action; a second
  pass targets thermal tasks. Labels by reset-and-replay BFS to depth 3 against a null
  baseline (see `docs/analysis_plan.md`).

## Counts

To be filled from `python -m irrev.dataset` output once labelling is final.

## Agreement

Two annotators label 100 random pairs blind to the automatic label
(`data/agreement/`). Cohen's kappa between annotators and with the automatic labels: to be
filled.

## Known limitations

- The sandbox is small and templated; reversibility is defined by its own design.
- ScienceWorld search is pruned (restoring verbs only, at most 10 children per node and 6
  expanded depth-2 nodes); pruned searches are flagged `truncated` and may mislabel a
  reversible action as irreversible.
- ScienceWorld heat transfer is not exactly reproducible across replays, so temperature
  tolerance depends on the observed replay drift.
- Partially reversible actions are rare in ScienceWorld under the structural definition.
- 3-step horizon.

## Licences

ScienceWorld: Apache-2.0 (LICENSE in the installed package `scienceworld==1.2.3`; re-check
the GitHub repository before release). Sandbox data: ours.
