"""OOM fallback in generate and length batching, with fake model and tokenizer objects."""

from __future__ import annotations

import pytest
import torch

from irrev.models import length_batches, split_on_oom


def test_split_on_oom_halves_until_batches_fit_and_keeps_order() -> None:
    calls: list[int] = []

    def run(batch: list[str]) -> list[str]:
        calls.append(len(batch))
        if len(batch) > 2:
            raise torch.OutOfMemoryError("fake")
        return [s.upper() for s in batch]

    assert split_on_oom(["a", "b", "c", "d", "e"], run) == ["A", "B", "C", "D", "E"]
    assert calls == [5, 2, 3, 1, 2]  # 5 -> 2 + 3, and 3 -> 1 + 2


def test_split_on_oom_reraises_for_a_single_item() -> None:
    def run(batch: list[int]) -> list[int]:
        raise torch.OutOfMemoryError("fake")

    with pytest.raises(torch.OutOfMemoryError):
        split_on_oom([1], run)


def test_length_batches_respect_budget() -> None:
    lengths = [100, 900, 300, 50, 800, 400]
    batches = length_batches(lengths, max_tokens=1000, max_batch=3)
    assert sorted(i for b in batches for i in b) == list(range(6))
    for b in batches:
        assert len(b) <= 3 and (len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 1000)


def test_llama_prompt_has_no_todays_date() -> None:
    from transformers import AutoTokenizer

    from irrev.models import FIXED_DATE, render

    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except OSError:
        pytest.skip("Llama tokenizer not available (gated)")
    text = render(tok, [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert f"Today Date: {FIXED_DATE}" in text
