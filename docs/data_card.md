# Data card: reversibility-labelled (state, action) pairs

File: `data/pairs.jsonl` (JSON lines). One record per (state, action) pair.

| Field | Meaning |
| --- | --- |
| `id` | unique id (`sb-...` sandbox, `sw-...` ScienceWorld, `toolemu_...` ToolEmu) |
| `environment` | `sandbox`, `scienceworld` or `toolemu` |
| `state_text` | the state as the model sees it |
| `action_text` | the candidate action (sandbox: one JSON tool call; ScienceWorld: one command) |
| `label` | `reversible`, `partially_reversible` or `irreversible` at horizon 3 |
| `horizon` | horizon of `label` (3) |
| `source` | `sandbox_design_rule`, `scienceworld_bfs_replay`, or the ToolEmu case it was adapted from |
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

2,648 pairs (target in the brief: at least 2,000). Minimal-pair rows: 1,442 (54.5%; brief
minimum 30%). The test split holds 20% of scenarios per environment (seeds 0, 1, 2).

| Environment | Pairs | Scenarios | Reversible | Partially reversible | Irreversible | Minimal-pair rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sandbox | 1,500 | 700 | 637 | 423 | 440 | 1,200 (80%) |
| ScienceWorld | 1,148 | 180 | 691 | 33 | 424 | 242 (21%) |
| total | 2,648 | 880 | 1,328 | 456 | 864 | 1,442 (54.5%) |

ScienceWorld labels by horizon (horizon 3 is `label`):

| Horizon | Reversible | Partially reversible | Irreversible |
| --- | ---: | ---: | ---: |
| 1 | 616 | 34 | 498 |
| 2 | 622 | 33 | 493 |
| 3 | 691 | 33 | 424 |

The classes are not balanced (partially reversible is rare in ScienceWorld), so every
probe result is reported as balanced accuracy, with macro-F1 and AUROC alongside.

ScienceWorld searches cut short by the pruning limits (`truncated` in
`data/raw/scienceworld_*.jsonl`): 580 of 1,148. A truncated search can only err towards
"irreversible" (a restoring sequence that was found is real), so 147 of the 424
irreversible labels rest on an incomplete search.

## ToolEmu cases in the sandbox

File: `data/toolemu_pairs.jsonl`, same fields, `environment = toolemu`. 54 of ToolEmu's
144 curated cases (assets/all_cases.json, Apache-2.0) whose risky action the sandbox models
exactly (deleting, overwriting, sending, paying) were rebuilt as paired sandbox tasks
(`irrev/toolemu.py`); each gives a reversible and an irreversible version of the same
action, so 108 pairs, 54 per class, all minimal pairs. Labels come from the sandbox design.
The other 90 cases (smart locks, robots, emergency dispatch, traffic control, trading
orders, public posts and similar) are listed with the reason in `data/toolemu_excluded.json`.
They are an external test set: no probe is trained on them.

## Agreement

Two annotators label 100 random pairs blind to the automatic label
(`data/agreement/annotator_A.csv`, `annotator_B.csv`; key in
`KEY_do_not_open_before_labelling.csv`). Cohen's kappa between the annotators and between
each annotator and the automatic label: written to `data/agreement/agreement.json` by
`python -m irrev.agreement`. Result (n = 100): kappa A vs B = 0.61 (raw agreement 0.77),
A vs automatic = 0.54, B vs automatic = 0.58.

## Known limitations

- The sandbox is small and templated; reversibility is defined by its own design.
- ScienceWorld search is pruned (restoring verbs only, at most 10 children per node and 6
  expanded depth-2 nodes); pruned searches are flagged `truncated` and may mislabel a
  reversible action as irreversible (147 of 424 irreversible labels).
- ScienceWorld heat transfer is not exactly reproducible across replays, so temperature
  tolerance depends on the observed replay drift.
- Partially reversible actions are rare in ScienceWorld under the structural definition.
- 3-step horizon.
- ToolEmu cases are adapted to four sandbox mechanisms; the wording keeps the ToolEmu
  instruction, but toolkit objects are replaced (a task, repository or booking becomes a
  file; an SMS, Slack or Gmail message becomes an email; a bank, crypto or Venmo payment or
  a purchase becomes a wallet transfer).

## Licences (checked 7 Nov 2026)

- ScienceWorld: Apache-2.0 (LICENSE in github.com/allenai/ScienceWorld and in the installed
  package `scienceworld==1.2.3`).
- ToolEmu curated cases: Apache-2.0 (LICENSE in github.com/ryoungj/ToolEmu); the adapted
  cases in `data/toolemu_pairs.jsonl` name their source case.
- ByteSized32-State-Prediction (RQ5 only): no licence file; used for research only and not
  redistributed (`data/rq5/` and `data/external/` are not committed).
- Sandbox data: ours.
