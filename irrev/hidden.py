"""Hidden states for every (state, action) pair at every layer (brief Step 2).

For each pair the prompt is the agent prompt (task, state, tool list) followed by the
candidate action as the assistant reply. One forward pass stores, for every layer
(index 0 = embeddings, index i = output of decoder layer i-1, last index after the
final norm), the hidden state of the last action token and the mean over action tokens.
Arrays are memory-mapped and filled batch by batch, so a restarted job resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from transformers import PreTrainedTokenizerBase

from irrev.dataset import Pair, read_pairs
from irrev.models import device, length_batches, load, n_layers, render, run_metadata
from irrev.prompts import pair_messages


def build_inputs(tokenizer: PreTrainedTokenizerBase, rows: list[Pair]) -> tuple[list[str], list[int]]:
    texts, starts = [], []
    for row in rows:
        prefix = render(tokenizer, pair_messages(row["environment"], row["state_text"], row["task_text"]))
        texts.append(prefix + row["action_text"])
        starts.append(len(prefix))
    return texts, starts


def action_token_mask(offsets: torch.Tensor, starts: list[int]) -> torch.Tensor:
    """True for tokens whose characters overlap the action text (pads have offset (0, 0))."""
    ends = offsets[..., 1]
    begin = torch.tensor(starts, device=offsets.device).unsqueeze(1)
    return (ends > begin) & (ends > offsets[..., 0])


def extract(model_name: str, pairs_path: Path, out_dir: Path, max_tokens: int, max_batch: int,
            store_dtype: str, limit: int) -> None:
    rows = read_pairs(pairs_path)
    if limit:
        rows = rows[:limit]
    model, tokenizer = load(model_name)
    layers, dim = n_layers(model) + 1, int(model.config.hidden_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path, done_path = out_dir / "ids.json", out_dir / "done.npy"
    ids = [r["id"] for r in rows]
    if ids_path.exists() and json.loads(ids_path.read_text()) != ids:
        raise RuntimeError(f"{out_dir} holds states for a different pair list; use a new directory")
    ids_path.write_text(json.dumps(ids))
    dtype = np.float16 if store_dtype == "float16" else np.float32
    mode = "r+" if done_path.exists() else "w+"
    arrays = {pos: open_memmap(out_dir / f"{pos}.npy", mode=mode, dtype=dtype, shape=(len(rows), layers, dim))
              for pos in ("last", "mean")}
    done = np.load(done_path) if done_path.exists() else np.zeros(len(rows), dtype=bool)
    meta = run_metadata(model_name) | {"layers_incl_embedding": str(layers), "hidden_size": str(dim),
                                       "store_dtype": store_dtype, "pairs": str(pairs_path)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    texts, starts = build_inputs(tokenizer, rows)
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    todo = [i for i in range(len(rows)) if not done[i]]
    batches = length_batches([lengths[i] for i in todo], max_tokens, max_batch)
    start_time = time.time()

    def states_for(idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """[L+1, B, D] last-token and mean-over-action states; halves the batch on CUDA OOM."""
        try:
            enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False,
                            return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            mask = action_token_mask(offsets, [starts[i] for i in idx]).to(device())
            if not bool(mask.any(dim=1).all()):
                raise RuntimeError(f"no action tokens found for rows {idx}")
            with torch.no_grad():
                hidden_states = model(**enc.to(device()), output_hidden_states=True).hidden_states  # L+1 x [B, T, D]
            rows_idx = torch.arange(len(idx), device=mask.device)
            last_pos = (mask * torch.arange(mask.shape[1], device=mask.device)).argmax(dim=1)
            weights = mask.to(hidden_states[0].dtype) / mask.sum(dim=1, keepdim=True)
            last = torch.stack([h[rows_idx, last_pos] for h in hidden_states])
            mean = torch.stack([torch.einsum("btd,bt->bd", h, weights) for h in hidden_states])
            return last, mean
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(idx) // 2
            print(f"OOM at batch {len(idx)}; retrying as {half} + {len(idx) - half}", flush=True)
            (l1, m1), (l2, m2) = states_for(idx[:half]), states_for(idx[half:])
            return torch.cat([l1, l2], dim=1), torch.cat([m1, m2], dim=1)

    for b, local in enumerate(batches, 1):
        idx = [todo[j] for j in local]
        last, mean = states_for(idx)
        for name, value in (("last", last), ("mean", mean)):
            arr = value.float().permute(1, 0, 2).cpu().numpy()
            if not np.isfinite(arr).all() or (dtype == np.float16 and np.abs(arr).max() >= 65000):
                raise RuntimeError(f"hidden states overflow {store_dtype}; rerun with --store-dtype float32")
            arrays[name][idx] = arr.astype(dtype)
        done[idx] = True
        for arr in arrays.values():
            arr.flush()
        np.save(done_path, done)
        print(f"batch {b}/{len(batches)} rows {int(done.sum())}/{len(rows)} {time.time() - start_time:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-layer hidden states for labelled pairs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--store-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    extract(args.model, args.pairs, args.out, args.max_tokens, args.max_batch, args.store_dtype, args.limit)


if __name__ == "__main__":
    main()
