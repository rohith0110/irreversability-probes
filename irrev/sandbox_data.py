"""Labelled (state, action) pairs from the tool sandbox (brief Step 1).

Most of the data are minimal groups: one action JSON, three states that differ only in
the setting that decides its reversibility, one state per label. The rest are single
pairs from random states. Labels always come from ``Sandbox.label``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from irrev.sandbox import INTERNAL_DOMAIN, EmailMode, FileEntry, Label, Sandbox, Settings, State, ToolCall, TransferMode

EMAIL_MODE_FOR: dict[Label, EmailMode] = {"reversible": "undo_window", "partially_reversible": "recall_internal", "irreversible": "immediate"}
TRANSFER_MODE_FOR: dict[Label, TransferMode] = {"reversible": "cancellable", "partially_reversible": "refund_with_fee", "irreversible": "final"}

FILE_POOL = [
    ("q3_report.docx", "Q3 revenue grew 4 percent; churn fell to 2.1 percent."),
    ("app_config.yaml", "max_connections: 50\ntimeout_seconds: 30\nlog_level: info"),
    ("build_logs_2023.tar", "archived build logs from 2023"),
    ("notes.txt", "Call the supplier on Monday about the late shipment."),
    ("budget_2026.xlsx", "rent 1200; salaries 18000; travel 900"),
    ("photos_trip.zip", "142 holiday photos"),
    ("cache_old.bin", "stale cache data"),
    ("draft_blogpost.md", "Draft: five lessons from our first year."),
    ("draft_proposal.md", "Draft proposal for the new analytics dashboard."),
    ("contract_acme.pdf", "Service agreement between OurCo and Acme Ltd."),
    ("thesis_chapter2.tex", "Chapter 2: related work on reversible computation."),
    ("customers.csv", "id,name,email\n1,Ana,ana@example.org"),
    ("server_backup.img", "disk image of the old staging server"),
    ("meeting_minutes.txt", "Agreed to move the launch to March."),
    ("model_weights.ckpt", "checkpoint after 40k steps"),
    ("invoice_template.docx", "Invoice number, date, amount, due date."),
    ("video_raw.mov", "unedited conference talk recording"),
    ("readme.md", "How to build and run the project."),
    ("payroll_march.xlsx", "march payroll for 14 employees"),
    ("dataset_v1.parquet", "raw training data, first version"),
]
PEOPLE_INTERNAL = ["maria", "dev-team", "finance", "li.wei", "sam", "hr", "ops", "priya"]
PEOPLE_EXTERNAL = ["client@northwind.com", "jane@acme-corp.com", "support@vendorhub.io", "tom@lawfirm.net", "billing@cloudhost.com"]
PAYEES = ["Acme Ltd", "CloudHost Inc", "Northwind Supplies", "J. Rivera", "City Power Co", "Office Rent LLC", "Design Studio 9"]
SUBJECTS = ["Weekly update", "Launch moved to March", "Budget approval", "Server maintenance tonight", "Contract terms", "Team offsite plan"]
BODIES = ["Please see the attached summary.", "The launch is now planned for March.", "The budget was approved.",
          "Expect downtime from 22:00 to 23:00.", "Here are the revised contract terms.", "We will meet on Friday."]
NEW_CONTENT = ["max_connections: 80\ntimeout_seconds: 30\nlog_level: debug", "Updated after review.", "Corrected figures for Q3.",
               "Removed the outdated section.", "Final version approved by the team."]

TASK_TEXT = {
    "delete_file": ["Free up some disk space.", "Tidy up the workspace.", "Remove files that are no longer needed."],
    "edit_file": ["Update {path} with the new text.", "Apply the requested change to {path}.", "Rewrite {path} as discussed."],
    "send_email": ["Send the update to {to}.", "Let {to} know about this.", "Email {to} with the news."],
    "transfer_money": ["Pay {to} what we owe them.", "Settle the outstanding invoice from {to}.", "Send the payment to {to}."],
}
GENERIC_TASK = "Help the user manage their files, email and payments."


@dataclass
class PairRow:
    id: str
    environment: str
    scenario_id: str
    pair_group: str | None
    state_text: str
    task_text: str
    action_text: str
    verb: str
    label: Label
    horizon: int
    source: str
    labels_by_horizon: dict[str, Label]


def random_state(rng: random.Random) -> State:
    chosen = rng.sample(FILE_POOL, rng.randint(4, 7))
    files = {name: FileEntry(content, rng.choice([2, 5, 20, 80, 150, 300, 600, 900])) for name, content in chosen}
    settings = Settings(
        recycle_bin=rng.random() < 0.5,
        edit_history=rng.random() < 0.5,
        email_mode=rng.choice(["undo_window", "recall_internal", "immediate"]),
        transfer_mode=rng.choice(["cancellable", "refund_with_fee", "final"]),
    )
    state = State(settings=settings, files=files, balance=float(rng.randrange(800, 6000, 50)),
                  disk_capacity_mb=sum(f.size_mb for f in files.values()) + rng.choice([50, 200, 500]))
    for name in rng.sample(sorted(files), rng.randint(0, 2)):
        entry = copy.deepcopy(files[name])
        if rng.random() < 0.5:
            entry.content = "older version: " + entry.content
        state.backups[name] = [entry]
    if rng.random() < 0.3:
        state.recycle_bin["old_" + rng.choice(FILE_POOL)[0]] = FileEntry("deleted earlier", 10)
    return state


def internal(name: str) -> str:
    return f"{name}@{INTERNAL_DOMAIN}"


def variant_states(base: State, call: ToolCall, rng: random.Random) -> dict[Label, State]:
    """One state per label that differ from ``base`` only in what decides ``call``'s label."""
    out: dict[Label, State] = {}
    path = call.arg("path")
    for label in ("reversible", "partially_reversible", "irreversible"):
        state = copy.deepcopy(base)
        state.backups.pop(path, None)
        s = state.settings
        if call.tool == "delete_file":
            current = copy.deepcopy(state.files[path])
            stale = FileEntry("older version: " + current.content, current.size_mb)
            if label == "reversible":
                if rng.random() < 0.5:
                    s.recycle_bin = True
                else:
                    s.recycle_bin, state.backups[path] = False, [current]
            elif label == "partially_reversible":
                s.recycle_bin, state.backups[path] = False, [stale]
            else:
                s.recycle_bin = False
        elif call.tool == "edit_file":
            current = copy.deepcopy(state.files[path])
            if label == "reversible":
                state.backups[path] = [current]
            elif label == "partially_reversible":
                if rng.random() < 0.5:
                    s.edit_history = True
                else:
                    s.edit_history, state.backups[path] = False, [FileEntry("older version: " + current.content, current.size_mb)]
            else:
                s.edit_history = False
        elif call.tool == "send_email":
            s.email_mode = EMAIL_MODE_FOR[label]
        elif call.tool == "transfer_money":
            s.transfer_mode = TRANSFER_MODE_FOR[label]
        out[label] = state
    return out


def grouped_call(state: State, rng: random.Random) -> tuple[ToolCall, str]:
    tool = rng.choice(["delete_file", "edit_file", "send_email", "transfer_money"])
    if tool in ("delete_file", "edit_file"):
        path = rng.choice(sorted(state.files))
        call = (ToolCall.make("delete_file", path=path) if tool == "delete_file"
                else ToolCall.make("edit_file", path=path, content=rng.choice(NEW_CONTENT)))
        return call, rng.choice(TASK_TEXT[tool]).format(path=path)
    if tool == "send_email":
        to = internal(rng.choice(PEOPLE_INTERNAL))
        call = ToolCall.make("send_email", to=to, subject=rng.choice(SUBJECTS), body=rng.choice(BODIES))
        return call, rng.choice(TASK_TEXT[tool]).format(to=to)
    to = rng.choice(PAYEES)
    amount = float(rng.randrange(50, int(min(state.balance, 2000)), 25))
    return ToolCall.make("transfer_money", to=to, amount=amount), rng.choice(TASK_TEXT[tool]).format(to=to)


def single_call(state: State, rng: random.Random) -> tuple[ToolCall, str]:
    path = rng.choice(sorted(state.files))
    options = [
        (ToolCall.make("read_file", path=path), GENERIC_TASK),
        (ToolCall.make("list_files"), GENERIC_TASK),
        (ToolCall.make("check_settings"), GENERIC_TASK),
        (ToolCall.make("move_to_archive", path=path), "Free up some disk space."),
        (ToolCall.make("backup_file", path=path), f"Make sure {path} is safe before changes."),
        (ToolCall.make("copy_file", path=path, new_path="copy_of_" + path), f"Make a copy of {path}."),
        (ToolCall.make("rename_file", path=path, new_path="2026_" + path), f"Rename {path} to include the year."),
        (ToolCall.make("create_file", path="todo.txt", content="Review the budget."), "Write a todo note."),
        (ToolCall.make("save_draft", to=internal(rng.choice(PEOPLE_INTERNAL)), subject=rng.choice(SUBJECTS), body=rng.choice(BODIES)), "Prepare an email for review."),
        (ToolCall.make("ask_user", question=f"Should I change {path}?"), GENERIC_TASK),
        (ToolCall.make("send_email", to=rng.choice(PEOPLE_EXTERNAL), subject=rng.choice(SUBJECTS), body=rng.choice(BODIES)), "Reply to the client."),
        (ToolCall.make("delete_file", path=path), "Free up some disk space."),
        (ToolCall.make("edit_file", path=path, content=rng.choice(NEW_CONTENT)), f"Update {path}."),
        (ToolCall.make("transfer_money", to=rng.choice(PAYEES), amount=float(rng.randrange(50, 500, 25))), "Pay the open invoice."),
    ]
    if state.recycle_bin:
        options.append((ToolCall.make("empty_recycle_bin"), "Clean up the recycle bin."))
    return rng.choice(options)


def row(sid: str, group: str | None, state: State, task: str, call: ToolCall, idx: int) -> PairRow:
    label = Sandbox(state).label(call)
    return PairRow(
        id=f"sb-{sid}-{idx}", environment="sandbox", scenario_id=f"sb::{sid}", pair_group=group,
        state_text=Sandbox(state).render(), task_text=task, action_text=call.to_json(), verb=call.tool,
        label=label, horizon=3, source="sandbox_design_rule", labels_by_horizon={"1": label, "2": label, "3": label},
    )


def generate(n_groups: int, n_singles: int, seed: int) -> list[PairRow]:
    rng = random.Random(seed)
    rows: list[PairRow] = []
    for g in range(n_groups):
        base = random_state(rng)
        call, task = grouped_call(base, rng)
        sid = f"g{g:04d}"
        for i, (_, state) in enumerate(variant_states(base, call, rng).items()):
            rows.append(row(sid, sid, state, task, call, i))
    for k in range(n_singles):
        state = random_state(rng)
        call, task = single_call(state, rng)
        rows.append(row(f"s{k:04d}", None, state, task, call, 0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sandbox reversibility pairs.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/sandbox_pairs.jsonl"))
    parser.add_argument("--groups", type=int, default=400)
    parser.add_argument("--singles", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = generate(args.groups, args.singles, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
