"""Model loading, chat-template rendering and run metadata shared by every GPU job."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Callable
from typing import TypeVar

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase

from irrev.prompts import Message

MODEL_ALIASES = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
}
SEED = 0
FIXED_DATE = "26 Jul 2024"


def resolve(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        # ponytail: fixed small pool; torch's default (all cores) thrashes when the laptop is shared
        torch.set_num_threads(int(os.environ.get("IRREV_CPU_THREADS", "4")))
    repo = resolve(name)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype)
    # In place; PreTrainedModel.to is typed through functools.wraps, which mypy cannot bind.
    # from_pretrained already puts the model in evaluation mode (dropout off).
    torch.nn.Module.to(model, device())
    return model, tokenizer


Item = TypeVar("Item")
Result = TypeVar("Result")


def split_on_oom(items: list[Item], run: Callable[[list[Item]], list[Result]]) -> list[Result]:
    """Run a batch; on CUDA out-of-memory split it in half and retry, down to single items."""
    try:
        return run(items)
    except torch.OutOfMemoryError:
        if len(items) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(items) // 2
        print(f"OOM at batch {len(items)}; retrying as {half} + {len(items) - half}", flush=True)
        return split_on_oom(items[:half], run) + split_on_oom(items[half:], run)


def render(tokenizer: PreTrainedTokenizerBase, messages: list[Message]) -> str:
    """Chat-template text ending with the assistant header, ready for generation or an action."""
    # Llama 3 templates stamp today's date into the system prompt; pin the template's own
    # default so prompts are identical on every day and machine (ignored by other templates).
    chat = [{"role": m["role"], "content": m["content"]} for m in messages]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE)
    if not isinstance(text, str):
        raise TypeError("apply_chat_template(tokenize=False) did not return text")
    return text


def n_layers(model: PreTrainedModel) -> int:
    return int(model.config.num_hidden_layers)


def run_metadata(model_name: str) -> dict[str, str]:
    """GPU, driver, CUDA, torch and transformers versions, written into every result file."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    driver = "none"
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            driver = f"unknown ({exc})"
    return {
        "model": resolve(model_name), "gpu": gpu, "driver": driver, "cuda": str(torch.version.cuda),
        "torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0],
        "platform": platform.platform(), "host": os.environ.get("IRREV_HOST", platform.node()),
    }


def generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str],
             max_new_tokens: int) -> list[str]:
    """Greedy decoding (temperature 0) for a left-padded batch; returns only the new text.

    On CUDA out-of-memory the batch is split in half and retried, down to single prompts.
    """
    if not isinstance(model, GenerationMixin):  # every AutoModelForCausalLM is one
        raise TypeError(f"{type(model).__name__} cannot generate text")
    generator = model

    def run(batch_prompts: list[str]) -> list[str]:
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device())
        with torch.no_grad():
            out = generator.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
        if not isinstance(out, torch.Tensor):
            raise TypeError("generate returned a structured output; expected token ids")
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        texts: list[str] = tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return texts

    return split_on_oom(prompts, run)


def length_batches(lengths: list[int], max_tokens: int, max_batch: int) -> list[list[int]]:
    """Group indices of similar length so each padded batch stays under a token budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        longest = max([lengths[j] for j in current] + [lengths[i]])
        if current and (longest * (len(current) + 1) > max_tokens or len(current) >= max_batch):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches
