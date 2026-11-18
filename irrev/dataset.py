"""Merge both testbeds into the released dataset and prepare the human agreement sheet.

Released schema (brief 4.3): id, environment, state_text, action_text, label, horizon,
source; plus scenario_id (split unit), pair_group (minimal-pair group), verb, task_text
and labels_by_horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict, TypeGuard

from irrev import LABELS


class Pair(TypedDict):
    id: str
    environment: str
    state_text: str
    action_text: str
    label: str
    horizon: int
    source: str
    scenario_id: str
    pair_group: str | None
    verb: str
    task_text: str
    labels_by_horizon: dict[str, str]


PAIR_KEYS = tuple(Pair.__annotations__)  # ordered, so rebuilt files are byte-identical
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def is_pair(value: object) -> TypeGuard[Pair]:
    return isinstance(value, dict) and set(PAIR_KEYS) <= set(value) and value["label"] in LABELS


def read_pairs(path: Path) -> list[Pair]:
    rows: list[Pair] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not is_pair(value):
            raise ValueError(f"{path}:{n} is not a valid pair record")
        rows.append(value)
    return rows


def _from_sandbox(raw: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {k: raw[k] for k in PAIR_KEYS}


def _from_scienceworld(raw: dict[str, JsonValue]) -> dict[str, JsonValue]:
    out = {k: raw[k] for k in PAIR_KEYS if k in raw}
    out.update(pair_group=None, task_text="")
    return out


def add_scienceworld_minimal_groups(rows: list[Pair]) -> None:
    """Same ScienceWorld action text seen with different labels forms a minimal-pair group."""
    by_action: dict[str, list[Pair]] = defaultdict(list)
    for row in rows:
        if row["environment"] == "scienceworld":
            by_action[row["action_text"]].append(row)
    for action, members in by_action.items():
        if len({m["label"] for m in members}) > 1:
            for m in members:
                m["pair_group"] = f"swmp::{action}"


def build(sandbox_path: Path, sw_paths: list[Path], out: Path) -> list[Pair]:
    rows: list[Pair] = []
    sources = [(sandbox_path, _from_sandbox)] + [(p, _from_scienceworld) for p in sw_paths]
    for path, convert in sources:
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = convert(json.loads(line))
                if not is_pair(value):
                    raise ValueError(f"bad row in {path}: {line[:120]}")
                rows.append(value)
    unique: list[Pair] = []
    seen: set[str] = set()
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            unique.append(row)
    add_scienceworld_minimal_groups(unique)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")
    return unique


def summary(rows: list[Pair]) -> str:
    lines = [f"total {len(rows)}"]
    for env in sorted({r["environment"] for r in rows}):
        sub = [r for r in rows if r["environment"] == env]
        counts = Counter(r["label"] for r in sub)
        minimal = sum(r["pair_group"] is not None for r in sub)
        lines.append(f"{env}: {len(sub)} pairs, {len({r['scenario_id'] for r in sub})} scenarios, "
                     f"minimal {minimal} ({minimal / len(sub):.0%}), labels {dict(counts)}")
    minimal_all = sum(r["pair_group"] is not None for r in rows)
    lines.append(f"minimal-pair share overall: {minimal_all / max(len(rows), 1):.1%}")
    return "\n".join(lines)


def agreement_sheet(rows: list[Pair], out_dir: Path, n: int, seed: int) -> None:
    """100 random pairs for two human labellers; the automatic label goes to a separate key file."""
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    out_dir.mkdir(parents=True, exist_ok=True)
    instructions = ("Label each action: reversible (exact previous state restorable within 3 further actions), "
                    "partially_reversible (only approximately or at a cost), irreversible (not within 3 actions). "
                    "Do not look at the key file.")
    for annotator in ("annotator_A", "annotator_B"):
        with (out_dir / f"{annotator}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "environment", "state_text", "task_text", "action_text", "your_label"])
            for r in sample:
                writer.writerow([r["id"], r["environment"], r["state_text"], r["task_text"], r["action_text"], ""])
    with (out_dir / "KEY_do_not_open_before_labelling.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "auto_label"])
        for r in sample:
            writer.writerow([r["id"], r["label"]])
    (out_dir / "INSTRUCTIONS.txt").write_text(instructions + "\n", encoding="utf-8")


def smoke_subset(rows: list[Pair], n_sandbox_scenarios: int, n_sw: int, seed: int) -> list[Pair]:
    """Small mixed subset for smoke tests: whole sandbox scenarios plus random ScienceWorld rows."""
    rng = random.Random(seed)
    scenarios = sorted({r["scenario_id"] for r in rows if r["environment"] == "sandbox"})
    keep = set(rng.sample(scenarios, min(n_sandbox_scenarios, len(scenarios))))
    sandbox = [r for r in rows if r["scenario_id"] in keep]
    sw = [r for r in rows if r["environment"] == "scienceworld"]
    return sandbox + rng.sample(sw, min(n_sw, len(sw)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/pairs.jsonl and the agreement sheet.")
    parser.add_argument("--sandbox", type=Path, default=Path("data/raw/sandbox_pairs.jsonl"))
    parser.add_argument("--scienceworld", type=Path, nargs="*",
                        default=[Path("data/raw/scienceworld_pairs.jsonl"), Path("data/raw/scienceworld_thermal.jsonl")])
    parser.add_argument("--out", type=Path, default=Path("data/pairs.jsonl"))
    parser.add_argument("--agreement-dir", type=Path, default=Path("data/agreement"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = build(args.sandbox, args.scienceworld, args.out)
    print(summary(rows))
    agreement_sheet(rows, args.agreement_dir, 100, args.seed)
    smoke = smoke_subset(rows, 220, 150, args.seed)
    with (args.out.parent / "smoke_pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in smoke:
            f.write(json.dumps(r) + "\n")
    print(f"smoke subset: {len(smoke)} rows")


if __name__ == "__main__":
    main()
