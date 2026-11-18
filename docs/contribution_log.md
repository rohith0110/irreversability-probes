# Individual contribution log

One paragraph per team member (brief section 8). Each member replaces `[Paragraph.]` and
writes, in their own words, what they did, using the work packages below. Each package
names the files that hold the work, so every claim can be checked.

## Anshika Mishra (SE23UCSE027)

[Paragraph.]

## Banoth Rohith Rathod (SE23UCSE034)

[Paragraph.]

## Gonuguntla Rusheel (SE23UCSE067)

[Paragraph.]

## Goyal Aditya (SE23UARI003)

[Paragraph.]

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
