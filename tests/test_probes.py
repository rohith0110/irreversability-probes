"""The steering layer is chosen once; later probe runs must reuse it, never re-select or overwrite it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from irrev import probes


def synthetic(tmp: Path, n: int = 80, layers: int = 4, dim: int = 6) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for i in range(n):
        label = "irreversible" if i % 2 else "reversible"
        labels.append(i % 2)
        rows.append({"id": f"sb-{i}", "environment": "sandbox", "state_text": "s", "action_text": "a", "label": label,
                     "horizon": 3, "source": "test", "scenario_id": f"sc{i // 2}", "pair_group": None, "verb": "delete_file",
                     "task_text": "t", "labels_by_horizon": {"1": label, "2": label, "3": label}})
    pairs = tmp / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    hidden = tmp / "hidden"
    hidden.mkdir()
    states = rng.normal(size=(n, layers, dim)).astype(np.float32)
    states[:, 2, 0] += 3.0 * np.array(labels)  # layer 2 separates the classes
    for pos in ("last", "mean"):
        np.save(hidden / f"{pos}.npy", states)
    np.save(hidden / "done.npy", np.ones(n, dtype=bool))
    (hidden / "ids.json").write_text(json.dumps([r["id"] for r in rows]))
    return pairs, hidden


def run_select_layer(monkeypatch: pytest.MonkeyPatch, pairs: Path, hidden: Path, out: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["probes", "select-layer", "--pairs", str(pairs), "--hidden", str(hidden), "--out", str(out)])
    probes.main()


def test_first_run_selects_and_later_runs_keep_the_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, hidden = synthetic(tmp_path)
    out = tmp_path / "probes"
    run_select_layer(monkeypatch, pairs, hidden, out)
    assert json.loads((out / "best_layer.json").read_text())["hidden_index"] == 2
    direction = np.load(out / "direction.npy")

    (out / "best_layer.json").write_text(json.dumps({"hidden_index": 1}))  # a choice made earlier, e.g. on the GPU node
    run_select_layer(monkeypatch, pairs, hidden, out)
    assert json.loads((out / "best_layer.json").read_text())["hidden_index"] == 1
    assert np.array_equal(np.load(out / "direction.npy"), direction)  # the steering direction is not recomputed
