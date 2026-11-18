"""ToolEmu cases in the sandbox: minimal pairs with exact labels, and both paths can succeed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from irrev.sandbox import Sandbox, ToolCall
from irrev.toolemu import INCLUDED, excluded, pair_rows, toolemu_tasks

CASES = Path("data/external/toolemu/all_cases.json")


def test_every_case_is_a_minimal_pair_with_opposite_labels() -> None:
    for task in toolemu_tasks():
        rev, irr = Sandbox(task.build("reversible")), Sandbox(task.build("irreversible"))
        assert (rev.label(task.target_call()), irr.label(task.target_call())) == ("reversible", "irreversible"), task.id
        differing = [a for a, b in zip(rev.render().splitlines(), irr.render().splitlines()) if a != b]
        assert len(differing) == 1, (task.id, differing)


def test_shortcut_and_safe_path_both_succeed() -> None:
    for task in toolemu_tasks():
        for version in ("reversible", "irreversible"):
            shortcut = Sandbox(task.build(version))
            assert not shortcut.execute(task.target_call()).startswith("Error"), task.id
            assert task.success.check(shortcut.state), (task.id, version, "shortcut")
            safe = Sandbox(task.build(version))
            safe.execute(ToolCall.make("ask_user", question="ok?"))
            if task.mechanism == "recycle_bin":
                safe.execute(ToolCall.make("move_to_archive", path=task.target_call().arg("path")))
            else:
                if task.mechanism == "backup":
                    safe.execute(ToolCall.make("backup_file", path=task.success.paths[0]))
                    assert safe.label(task.target_call()) == "reversible", task.id
                safe.execute(task.target_call())
            assert task.success.check(safe.state), (task.id, version, "safe")


def test_pairs_are_balanced_and_unique() -> None:
    rows = pair_rows()
    assert len(rows) == 2 * len(toolemu_tasks())
    assert len({r["id"] for r in rows}) == len(rows)
    assert sum(r["label"] == "irreversible" for r in rows) == len(rows) // 2


def test_included_and_excluded_cover_the_144_curated_cases() -> None:
    if not CASES.exists():
        pytest.skip("ToolEmu all_cases.json not downloaded (data/external is not committed)")
    names = {c["name"] for c in json.loads(CASES.read_text(encoding="utf-8"))}
    skipped = {e["name"] for e in excluded(CASES)}
    assert len(names) == 144 and INCLUDED <= names and skipped == names - INCLUDED
