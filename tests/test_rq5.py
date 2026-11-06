"""RQ5 labelling on one ByteSized32 game (skipped when the unlicensed data are absent)."""

from __future__ import annotations

import json

import pytest

from irrev import rq5


pytestmark = pytest.mark.skipif(not rq5.TEST_FILE.exists(), reason="ByteSized32 data not downloaded")


def test_lightbulb_labels_are_consistent() -> None:
    records = [json.loads(line) for line in rq5.TEST_FILE.read_text(encoding="utf-8").splitlines()]
    mine = [r for r in records if r["game"] == "lit-lightbulb"]
    name, transitions, stats = rq5.label_game(("lit-lightbulb", mine))
    assert "unmatched_prefix" not in stats and transitions
    for t in transitions:
        order = ["reversible", "partially_reversible", "irreversible"]
        ranks = [order.index(t.labels_by_horizon[h]) for h in ("1", "2", "3")]
        assert ranks == sorted(ranks, reverse=True)  # a longer horizon never makes an action less reversible
        if t.label == "reversible":
            assert t.restore and len(t.restore) <= 3


def test_same_change_ignores_order() -> None:
    gold = json.dumps([{"name": "a", "contains": ["x", "y"]}, {"name": "b", "contains": []}])
    assert rq5.same_change('[{"name": "b", "contains": []}, {"name": "a", "contains": ["y", "x"]}]', gold)
    assert not rq5.same_change("[]", gold)
    assert not rq5.same_change("not json", gold)
