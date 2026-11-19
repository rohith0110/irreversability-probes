# Individual contribution log

One paragraph per team member (brief section 8). All four of us read the Group A papers. The
table at the end lists the files behind each piece of work.

## Anshika Mishra (SE23UCSE027)

I designed and built the tool sandbox: a Python class with files, a recycle bin, an external
archive, backups, an email outbox and a wallet, with 24 tools whose reversibility is fixed by
design. I wrote the breadth-first restore search that checks those labels in the tests. From the
sandbox I generated the 1,500 (state, action) pairs and worked out the minimal-pair scheme, where
the same tool call sits in two states that differ only in the setting that decides whether it can
be undone. I also wrote the 60 paired behaviour tasks across nine families, each with a shortcut
that is irreversible only in one version and a safer path in both, plus the code that turns agent
runs into caution rates, the blindness gap and the McNemar tests. The RQ2 prompts, which ask the
model directly whether an action can be undone, are mine as well.

## Banoth Rohith Rathod (SE23UCSE034)

I built the parts of the pipeline that touch the models, starting with the ScienceWorld labeller.
ScienceWorld has no save and restore, so the labeller rebuilds each state by resetting the task
variation and replaying its action history, then searches for restoring sequences of up to three
actions. My first version matched objects by name and treated replay noise as real change, which
gave false "irreversible" labels; I rewrote it and re-ran the labelling from scratch. I then wrote
the hidden-state extraction and the probe code with its controls (lexical baseline, shuffled
labels, the Hewitt-Liang control task, layer 0), the ablations and the bootstrap intervals. On the
agent side I wrote the agent loop, the caution bookkeeping and the steering hook, along with the
rule that sets the steering strength range from parse-failure rates. I ran all three models on
Modal L40S GPUs through a job queue and wrote the run book. When a re-run picked neighbouring
layers and overwrote the 14B steering files, I tracked it down, set the affected runs aside, re-ran
them on the right layer and changed the analysis so it never re-selects a layer that already exists.

## Gonuguntla Rusheel (SE23UCSE067)

I merged the sandbox and ScienceWorld pairs into the final 2,648-pair dataset, made the
scenario-level splits for three seeds and wrote the data card. I ran the human agreement check:
100 random pairs, label sheets that hide the automatic label, and Cohen's kappa between the two
annotators and against the automatic labels. I went through the 144 curated ToolEmu cases, picked
the 54 whose risky action the sandbox models exactly, and rebuilt each one as a reversible and an
irreversible task. I also made the figures for the report.

## Goyal Aditya (SE23UARI003)

I handled the reading behind the Related Work section: the Group B summaries, an arXiv search for
newer work on risky agent actions and reversibility, and the bibliography. I wrote the analysis
plan before we had any test results and kept the log of what we added after it. For RQ5 I
labelled the ByteSized32 game transitions with the same restore search, using deep copies of the
game for exact save and restore, and measured how uncertain each model is about the next state. I
made the presentation slides and recorded the demo.

## Work packages

| Package | Files |
| --- | --- |
| Background reading (Group A for everyone; Group B summaries) and the arXiv search for newer work | `report/references.bib`, Related Work in `report/main.tex` |
| Tool sandbox, reversibility design rules, restore-search verification | `irrev/sandbox.py`, `tests/test_sandbox.py` |
| Sandbox (state, action) pairs and minimal-pair design | `irrev/sandbox_data.py` |
| ScienceWorld labelling by reset-and-replay BFS, thermal pass | `irrev/sw_label.py`, `data/raw/` |
| Dataset merge, splits, data card | `irrev/dataset.py`, `docs/data_card.md` |
| Human agreement labelling (100 pairs, two annotators) and Cohen's kappa | `data/agreement/`, `irrev/agreement.py` |
| 60 paired behaviour tasks | `irrev/tasks.py`, `tests/test_tasks.py` |
| ToolEmu cases rebuilt in the sandbox | `irrev/toolemu.py`, `data/toolemu_pairs.jsonl` |
| Hidden-state extraction | `irrev/hidden.py` |
| Probes, controls, ablations, confidence intervals | `irrev/probes.py` |
| RQ2 stated answers | `irrev/stated.py`, `irrev/prompts.py` |
| Agent loop, caution bookkeeping, steering hook and calibration | `irrev/agent.py`, `tests/test_agent.py` |
| Behaviour metrics, blindness gap, McNemar tests | `irrev/behaviour.py`, `irrev/stats.py` |
| RQ5 ByteSized32 labelling and uncertainty | `irrev/rq5.py` |
| GPU runs (Modal L40S), job queue, run book | `modal_app.py`, `irrev/run_queue.py`, `docs/node_runbook.md` |
| Analysis plan (pre-registration) and change log | `docs/analysis_plan.md` |
| Figures, tables and the paper | `irrev/figures.py`, `irrev/paper.py`, `report/` |
| Presentation and recorded demo | slides, `irrev/demo.py`, `demo/` |
