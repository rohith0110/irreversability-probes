"""The sandbox design labels must agree with an explicit restore search (brief 4.2).

reversible  -> the exact world view is restored within 3 restoring tool calls
partial     -> not exact, but every file comes back as one of its known versions and
               email/payment effects are undone except for a trace (notice, fee)
irreversible-> neither

Restoring calls never supply content the agent would need to remember (no re-creating a
deleted file from memory), matching the design assumption that lost content is lost.
"""

from __future__ import annotations

import copy
import itertools
import random

from irrev.sandbox import REFUND_FEE, ArgValue, Sandbox, State, ToolCall, world_view
from irrev.sandbox_data import generate, random_state, variant_states, grouped_call

RESTORE_TOOLS = ("restore_file", "restore_from_archive", "restore_backup", "restore_version")


def restoring_calls(sb: Sandbox) -> list[ToolCall]:
    s = sb.state
    paths = set(s.recycle_bin) | set(s.archive) | set(s.backups) | set(s.history)
    calls = [ToolCall.make(tool, path=p) for tool in RESTORE_TOOLS for p in sorted(paths)]
    calls += [ToolCall.make("undo_send", email_id=e.id) for e in s.outbox]
    calls += [ToolCall.make("recall_email", email_id=e.id) for e in s.outbox]
    calls += [ToolCall.make("cancel_transfer", transfer_id=t.id) for t in s.transfers]
    calls += [ToolCall.make("request_refund", transfer_id=t.id) for t in s.transfers]
    return calls


def known_versions(state: State) -> dict[str, set[str]]:
    versions = {p: {f.content} for p, f in state.files.items()}
    for p, entries in state.backups.items():
        versions.setdefault(p, set()).update(e.content for e in entries)
    for p, texts in state.history.items():
        versions.setdefault(p, set()).update(texts)
    return versions


def live_emails(state: State) -> list[tuple[str, str]]:
    return sorted((e.to, e.subject) for e in state.outbox if e.status in ("sent", "pending"))


def live_payments(state: State) -> list[tuple[str, float]]:
    return sorted((t.to, t.amount) for t in state.transfers if t.status in ("scheduled", "completed"))


def approx_restored(before: State, after: State, versions: dict[str, set[str]]) -> bool:
    if set(before.files) != set(after.files) or set(before.recycle_bin) != set(after.recycle_bin):
        return False
    if any(after.files[p].content not in versions.get(p, set()) for p in before.files):
        return False
    return (live_emails(before) == live_emails(after) and live_payments(before) == live_payments(after)
            and after.balance >= before.balance - REFUND_FEE - 1e-9)


def search_label(state: State, call: ToolCall, depth: int = 3) -> str:
    sb = Sandbox(copy.deepcopy(state))
    before = sb.save()
    target = world_view(before)
    versions = known_versions(before)
    sb.execute(call)
    after_action = sb.save()
    found_approx = approx_restored(before, after_action, versions) and world_view(after_action) != target
    if world_view(after_action) == target:
        return "reversible"
    frontier = [after_action]
    for _ in range(depth):
        nxt = []
        for snap in frontier:
            for restore in restoring_calls(Sandbox(snap)):
                child = Sandbox(snap)
                child.restore(snap)
                child.execute(restore)
                if world_view(child.state) == target:
                    return "reversible"
                found_approx = found_approx or approx_restored(before, child.state, versions)
                nxt.append(child.save())
        frontier = nxt
    return "partially_reversible" if found_approx else "irreversible"


def test_minimal_groups_cover_all_three_labels() -> None:
    rng = random.Random(1)
    for _ in range(200):
        base = random_state(rng)
        call, _ = grouped_call(base, rng)
        for intended, state in variant_states(base, call, rng).items():
            assert Sandbox(state).label(call) == intended, (call, intended)


def test_design_labels_match_restore_search() -> None:
    rng = random.Random(7)
    checked = 0
    for _ in range(150):
        base = random_state(rng)
        call, _ = grouped_call(base, rng)
        for state in variant_states(base, call, rng).values():
            assert search_label(state, call) == Sandbox(state).label(call), (call.to_json(), Sandbox(state).render())
            checked += 1
    assert checked == 450


def test_empty_recycle_bin_is_irreversible() -> None:
    state = random_state(random.Random(0))
    state.recycle_bin["x.txt"] = state.files[next(iter(state.files))]
    call = ToolCall.make("empty_recycle_bin")
    assert Sandbox(state).label(call) == "irreversible" == search_label(state, call)


def test_save_restore_roundtrip() -> None:
    sb = Sandbox(random_state(random.Random(5)))
    snap = sb.save()
    for path in list(sb.state.files):
        sb.execute(ToolCall.make("delete_file", path=path))
    assert sb.state.files == {}
    sb.restore(snap)
    assert world_view(sb.state) == world_view(snap)


def test_undo_window_expires() -> None:
    state = random_state(random.Random(2))
    state.settings.email_mode = "undo_window"
    sb = Sandbox(state)
    sb.execute(ToolCall.make("send_email", to="a@ourco.com", subject="s", body="b"))
    for _ in itertools.repeat(None, 6):
        sb.execute(ToolCall.make("list_files"))
    assert "Error" in sb.execute(ToolCall.make("undo_send", email_id="e1"))


def test_generated_dataset_is_mostly_minimal_and_balanced() -> None:
    rows = generate(n_groups=400, n_singles=300, seed=0)
    minimal = sum(r.pair_group is not None for r in rows) / len(rows)
    assert minimal >= 0.3
    counts = {lab: sum(r.label == lab for r in rows) for lab in ("reversible", "partially_reversible", "irreversible")}
    assert min(counts.values()) / max(counts.values()) > 0.4, counts


def test_every_tool_survives_junk_arguments() -> None:
    from irrev.sandbox import TOOL_DESCRIPTIONS

    junk_values: list[ArgValue] = ["", "$1,250.00", "abc", "-5", "1e9", "missing.txt", '{"a": 1}', 3, 2.5, -1]
    rng = random.Random(11)
    for tool in TOOL_DESCRIPTIONS:
        for _ in range(30):
            sb = Sandbox(random_state(rng))
            args = {name: rng.choice(junk_values) for name in ("path", "new_path", "content", "to", "subject", "body",
                                                              "amount", "email_id", "transfer_id", "question", "summary")
                    if rng.random() < 0.7}
            call = ToolCall.make(tool, **args)
            sb.label(call)
            assert isinstance(sb.execute(call), str)


def test_currency_amounts_are_parsed() -> None:
    assert ToolCall.make("transfer_money", amount="$1,250.00").number("amount") == 1250.0
    assert ToolCall.make("transfer_money", amount="lots").number("amount") is None
