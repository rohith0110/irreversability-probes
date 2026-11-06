"""Cohen's kappa for the 100-pair human labelling (brief 4.2): annotator A vs B, and each vs the automatic label.

Usage: python -m irrev.agreement --dir data/agreement
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from irrev import LABELS

ALIASES = {"reversible": "reversible", "r": "reversible", "partially_reversible": "partially_reversible",
           "partially reversible": "partially_reversible", "partial": "partially_reversible", "p": "partially_reversible",
           "irreversible": "irreversible", "i": "irreversible"}


def read_labels(path: Path, column: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as f:
        out = {}
        for row in csv.DictReader(f):
            value = ALIASES.get(row[column].strip().lower())
            if value is None:
                raise ValueError(f"{path}: id {row['id']} has label {row[column]!r}; use one of {LABELS}")
            out[row["id"]] = value
        return out


def kappas(a: dict[str, str], b: dict[str, str], auto: dict[str, str]) -> dict[str, float]:
    ids = sorted(set(a) & set(b) & set(auto))
    pairs = {"A_vs_B": (a, b), "A_vs_auto": (a, auto), "B_vs_auto": (b, auto)}
    out = {name: float(cohen_kappa_score([x[i] for i in ids], [y[i] for i in ids], labels=list(LABELS)))
           for name, (x, y) in pairs.items()}
    out["n"] = float(len(ids))
    out["raw_agreement_A_vs_B"] = sum(a[i] == b[i] for i in ids) / max(1, len(ids))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=Path("data/agreement"))
    args = parser.parse_args()
    a = read_labels(args.dir / "annotator_A.csv", "your_label")
    b = read_labels(args.dir / "annotator_B.csv", "your_label")
    auto = read_labels(args.dir / "KEY_do_not_open_before_labelling.csv", "auto_label")
    result = kappas(a, b, auto)
    (args.dir / "agreement.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
