"""Prompt parsing and the judge prompt's separation from the agent instructions."""

from irrev.prompts import parse_stated, stated_messages


def test_parse_stated() -> None:
    assert parse_stated("Irreversible.") == "irreversible"
    assert parse_stated("partially reversible") == "partially_reversible"
    assert parse_stated("Reversible - it can be undone") == "reversible"
    assert parse_stated("This cannot be undone") == "irreversible"
    assert parse_stated("not reversible") == "irreversible"
    assert parse_stated('{"tool": "x"}') == "unparsed"


def test_judge_prompt_has_no_tool_call_instruction() -> None:
    msgs = stated_messages("sandbox", "Files: a", "Do it", '{"tool": "delete_file", "args": {"path": "a"}}')
    text = " ".join(m["content"] for m in msgs)
    assert "Reply with exactly one tool call" not in text and "irreversible" in text


def test_option_order_counterbalanced() -> None:
    from irrev.stated import reverse_order

    fwd = stated_messages("sandbox", "s", "t", "a", False)[1]["content"]
    rev = stated_messages("sandbox", "s", "t", "a", True)[1]["content"]
    assert fwd.index("- reversible:") < fwd.index("- irreversible:")
    assert rev.index("- irreversible:") < rev.index("- reversible:")
    flips = [reverse_order(f"id{i}") for i in range(400)]
    assert 150 < sum(flips) < 250


def test_toolemu_pairs_use_the_sandbox_prompts() -> None:
    """ToolEmu cases run inside the sandbox; a ScienceWorld prompt here would corrupt their hidden states."""
    from irrev.prompts import pair_messages, stated_messages

    assert pair_messages("toolemu", "s", "t") == pair_messages("sandbox", "s", "t")
    assert stated_messages("toolemu", "s", "t", "a") == stated_messages("sandbox", "s", "t", "a")
    assert pair_messages("scienceworld", "s", "t") != pair_messages("sandbox", "s", "t")
