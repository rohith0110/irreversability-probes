"""RQ2: ask the model directly whether each action can be undone (brief Step 4).

Two inputs: every labelled pair, and the target action of every task version (the K
term of the blindness gap). Answers are parsed into the three labels; unparsed answers
are kept and counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from irrev.dataset import read_pairs
from irrev.models import generate, length_batches, load, render, run_metadata
from irrev.prompts import Message, parse_stated, stated_messages
from irrev.sandbox import Sandbox
from irrev.tasks import all_tasks

MAX_NEW_TOKENS = 10


def reverse_order(ident: str) -> bool:
    """Stable per-item coin flip for the option order."""
    return hashlib.sha1(ident.encode()).digest()[0] % 2 == 1


def questions(pairs_path: Path, limit: int) -> list[tuple[str, dict[str, str], list[Message]]]:
    items: list[tuple[str, dict[str, str], list[Message]]] = []
    rows = read_pairs(pairs_path)
    for row in rows[:limit] if limit else rows:
        flip = reverse_order(row["id"])
        info = {"kind": "pair", "label": row["label"], "environment": row["environment"], "options_reversed": str(flip)}
        items.append((row["id"], info, stated_messages(row["environment"], row["state_text"], row["task_text"],
                                                       row["action_text"], flip)))
    for task in all_tasks():
        for version in ("reversible", "irreversible"):
            sb = Sandbox(task.build(version))
            call = task.target_call()
            ident = f"task::{task.id}::{version}"
            flip = reverse_order(ident)
            info = {"kind": "task", "task_id": task.id, "version": version, "label": sb.label(call),
                    "environment": "sandbox", "options_reversed": str(flip)}
            items.append((ident, info, stated_messages("sandbox", sb.render(), task.instruction, call.to_json(), flip)))
    return items


def run(model_name: str, pairs_path: Path, out_path: Path, max_tokens: int, max_batch: int, limit: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}
    items = [it for it in questions(pairs_path, limit) if it[0] not in done]
    if not items:
        print("stated: nothing to do")
        return
    model, tokenizer = load(model_name)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(run_metadata(model_name) | {"max_new_tokens": str(MAX_NEW_TOKENS)}, indent=2))
    prompts = [render(tokenizer, msgs) for _, _, msgs in items]
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    start = time.time()
    batches = length_batches(lengths, max_tokens, max_batch)
    with out_path.open("a", encoding="utf-8") as out:
        for b, idx in enumerate(batches, 1):
            answers = generate(model, tokenizer, [prompts[i] for i in idx], MAX_NEW_TOKENS)
            for i, answer in zip(idx, answers):
                ident, info, _ = items[i]
                out.write(json.dumps({"id": ident, **info, "prompt": prompts[i], "raw": answer,
                                      "stated": parse_stated(answer)}) + "\n")
            out.flush()
            print(f"stated batch {b}/{len(batches)} {time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 stated reversibility answers.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args.model, args.pairs, args.out, args.max_tokens, args.max_batch, args.limit)


if __name__ == "__main__":
    main()
