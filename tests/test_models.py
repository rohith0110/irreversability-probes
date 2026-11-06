"""OOM fallback in generate and length batching, with fake model and tokenizer objects."""

from __future__ import annotations

import torch

from irrev.models import generate, length_batches


class FakeBatch(dict):
    def to(self, _device: torch.device) -> "FakeBatch":
        return self


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompts: list[str], **_: bool | str) -> FakeBatch:
        return FakeBatch(input_ids=torch.zeros((len(prompts), 3), dtype=torch.long))

    def batch_decode(self, tokens: torch.Tensor, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> list[str]:
        return [str(int(row[0])) for row in tokens]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def generate(self, input_ids: torch.Tensor, **_: int | bool | None) -> torch.Tensor:
        self.calls.append(len(input_ids))
        if len(input_ids) > 2:
            raise torch.OutOfMemoryError("fake")
        new = torch.arange(len(input_ids)).unsqueeze(1) + 7
        return torch.cat([input_ids, new], dim=1)


def test_generate_halves_on_oom() -> None:
    model = FakeModel()
    out = generate(model, FakeTokenizer(), ["a", "b", "c", "d", "e"], 4)
    assert len(out) == 5
    assert model.calls == [5, 2, 3, 1, 2]  # 5 -> 2 + 3, and 3 -> 1 + 2


def test_length_batches_respect_budget() -> None:
    lengths = [100, 900, 300, 50, 800, 400]
    batches = length_batches(lengths, max_tokens=1000, max_batch=3)
    assert sorted(i for b in batches for i in b) == list(range(6))
    for b in batches:
        assert len(b) <= 3 and (len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 1000)


def test_llama_prompt_has_no_todays_date() -> None:
    import pytest
    from transformers import AutoTokenizer

    from irrev.models import FIXED_DATE, render

    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except OSError:
        pytest.skip("Llama tokenizer not available (gated)")
    text = render(tok, [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert f"Today Date: {FIXED_DATE}" in text
