"""RQ5: is the model's uncertainty about the next state related to reversibility?

Data: ByteSized32-State-Prediction test split (cognitiveailab/GPT-simulator). No licence
file is published, so the data and game code stay in ``data/external`` (git-ignored) and
are never redistributed.

label   (CPU, Python >= 3.12 because some game files use 3.12 f-strings; stdlib only)
        Each test record is matched to the gold-playthrough prefix that produces its
        current state; the transition is labelled by breadth-first search to depth 3
        with the same definition as ScienceWorld (null baseline of 0..4 "look around"
        ticks; structure + numbers equal = reversible, structure equal = partially
        reversible, else irreversible). The games are pure Python, so ``copy.deepcopy``
        is an exact save/restore. Transitions that change nothing are not labelled.
nll     (GPU) The model predicts which objects change; we record the mean token negative
        log-likelihood of its own greedy prediction (primary, as the brief asks) and of
        the gold change list under teacher forcing (secondary).
analyze (CPU) AUROC of each uncertainty for separating irreversible from reversible
        transitions, with a 1,000-resample bootstrap CI, per-class means and a
        Mann-Whitney test.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict

GAME_DIR = Path("data/external/bytesized32/src/data/games")
PLAYTHROUGH_DIR = Path("data/external/bytesized32/src/data/playthroughs")
TEST_FILE = Path("data/external/bytesized32/test/test.jsonl")
MAX_DEPTH = 3
K_CHILDREN = 20
M_EXPAND = 8
NOOP = "look around"
SKIP_VERBS = {"look", "inventory", "examine", "read", "look around"}
INVERSE_VERBS = {
    "take": {"put"}, "put": {"take"}, "open": {"close"}, "close": {"open"}, "turn on": {"turn off"},
    "turn off": {"turn on"}, "activate": {"deactivate"}, "deactivate": {"activate"},
    "connect": {"disconnect"}, "disconnect": {"connect"}, "wear": {"take off", "remove"},
}


def verb_of(action: str) -> str:
    words = action.split(" ")
    two = " ".join(words[:2])
    return two if two in INVERSE_VERBS or two in ("take off",) else words[0]

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class GameObject(Protocol):
    name: str
    uuid: int
    properties: dict[str, JsonValue]
    contains: list["GameObject"]

    def getAllContainedObjectsRecursive(self) -> list["GameObject"]: ...


class TextGame(Protocol):
    rootObject: GameObject
    possibleActions: dict[str, list[list[str | GameObject]]]

    def generatePossibleActions(self) -> dict[str, list[list[str | GameObject]]]: ...

    def step(self, action: str) -> tuple[str, int, int, bool, bool]: ...

    def getTaskDescription(self) -> str: ...


class StateObject(TypedDict):
    name: str
    type: str
    properties: dict[str, JsonValue]
    contains: list[str]


class Record(TypedDict):
    current_state: dict[str, JsonValue]
    tick_state: dict[str, JsonValue]
    game: str
    state_id: int


@dataclass
class Transition:
    id: str
    game: str
    state_id: int
    prefix_len: int
    action: str
    task: str
    current_objects: str
    gold_change: str
    depth_full: int | None
    depth_struct: int | None
    restore: list[str] | None
    nodes: int
    truncated: bool
    labels_by_horizon: dict[str, str]
    label: str


def load_game(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"bs32_{name.replace('-', '_')}", GAME_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load game {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gold_actions(name: str) -> list[str]:
    lines = (PLAYTHROUGH_DIR / f"{name}-playthrough.txt").read_text(encoding="utf-8").splitlines()
    return [line[1:].strip() for line in lines if line.startswith(">") and line[1:].strip() not in ("help", "")]


def object_list(game: TextGame, name: str) -> list[StateObject]:
    """The objects the dataset records (``get_game_states.get_state``), without uuids."""
    objs = [] if name in ("metal-detector", "space-walk", "sunburn") else [game.rootObject]
    objs.extend(game.rootObject.getAllContainedObjectsRecursive())
    return [{"name": o.name, "type": type(o).__name__, "properties": copy.deepcopy(o.properties),
             "contains": [c.name for c in o.contains]} for o in objs]


def normalise(objects: list[StateObject]) -> list[StateObject]:
    """JSON round trip (tuples become lists), uuids dropped, contents sorted, for comparison."""
    clean: list[StateObject] = json.loads(json.dumps(
        [{k: v for k, v in o.items() if k not in ("uuid",)} for o in objects], default=str))
    for o in clean:
        o["contains"] = sorted(o["contains"])  # insertion order is not part of the state
    return sorted(clean, key=lambda o: json.dumps(o, sort_keys=True))


def split_numeric(objects: list[StateObject]) -> tuple[str, list[float]]:
    """Structure (every non-numeric property) and the numbers, in a fixed order."""
    numbers: list[float] = []
    struct = []
    for o in objects:
        props: dict[str, JsonValue] = {}
        for key, value in sorted(o["properties"].items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers.append(float(value))
            else:
                props[key] = value
        struct.append({"name": o["name"], "type": o["type"], "contains": o["contains"], "properties": props})
    return json.dumps(struct, sort_keys=True), numbers


class Snapshot:
    """Exact save/restore: the game object graph plus the module's global UUID counter."""

    def __init__(self, module: ModuleType, game: TextGame) -> None:
        self.module = module
        self.game = copy.deepcopy(game)
        self.uuid = module.UUID

    def resume(self) -> TextGame:
        self.module.UUID = self.uuid
        game: TextGame = copy.deepcopy(self.game)
        game.generatePossibleActions()
        return game


def changed_names(a: list[StateObject], b: list[StateObject]) -> tuple[set[str], set[str]]:
    """(moved or modified objects, containers whose contents changed) between two states."""
    index_a = {o["name"]: o for o in normalise(a)}
    index_b = {o["name"]: o for o in normalise(b)}
    moved: set[str] = set()
    containers: set[str] = set()
    for name in set(index_a) | set(index_b):
        oa, ob = index_a.get(name), index_b.get(name)
        if oa is None or ob is None:
            moved.add(name)
            continue
        if oa["properties"] != ob["properties"] or oa["type"] != ob["type"]:
            moved.add(name)
        if oa["contains"] != ob["contains"]:
            containers.add(name)
            moved.update(set(oa["contains"]) ^ set(ob["contains"]))
    return moved, containers


def candidates(game: TextGame, changed: tuple[set[str], set[str]], last_verb: str) -> tuple[list[str], bool]:
    """Actions whose first object changed or that involve a changed container; changed
    patients first, then verbs that invert the last action."""
    moved, containers = changed
    inverse = INVERSE_VERBS.get(last_verb, set())
    scored: list[tuple[int, int, int, str]] = []
    for action, arg_lists in game.possibleActions.items():
        if action.split(" ")[0] in SKIP_VERBS:
            continue
        objs = [getattr(x, "name", "") for x in arg_lists[0][1:] if not isinstance(x, str)]
        if not objs:
            continue
        rank = 0 if objs[0] in moved else 1 if objs[0] in containers else 2
        hits = sum(o in moved or o in containers for o in objs)
        if rank < 2 or hits:
            scored.append((rank, 0 if verb_of(action) in inverse else 1, -hits, action))
    scored.sort()
    return [a for *_, a in scored[:K_CHILDREN]], len(scored) > K_CHILDREN


def label_transition(module: ModuleType, start: Snapshot, action: str, name: str) -> tuple[int | None, int | None, list[str] | None, int, bool] | None:
    nulls: list[tuple[str, list[float]]] = []
    game = start.resume()
    nulls.append(split_numeric(normalise(object_list(game, name))))
    for _ in range(MAX_DEPTH + 1):
        game.step(NOOP)
        game.generatePossibleActions()
        nulls.append(split_numeric(normalise(object_list(game, name))))

    def match(objects: list[StateObject]) -> tuple[bool, bool]:
        struct, numbers = split_numeric(normalise(objects))
        same = [n for n in nulls if n[0] == struct]
        return bool(same), any(len(n[1]) == len(numbers) and all(abs(x - y) < 1e-6 for x, y in zip(n[1], numbers)) for n in same)

    game = start.resume()
    before = object_list(game, name)
    game.step(action)
    game.generatePossibleActions()
    after = object_list(game, name)
    struct_ok, _ = match(after)
    if struct_ok:
        return None  # no structural effect: not labelled
    depth_full: int | None = None
    depth_struct: int | None = None
    restore: list[str] | None = None
    root = Snapshot(module, game)
    cands, truncated = candidates(game, changed_names(after, before), verb_of(action))
    level: list[tuple[list[str], Snapshot, list[str], int]] = [([], root, cands, 0)]
    seen = {json.dumps(normalise(after), sort_keys=True)}
    nodes = 0
    for depth in range(1, MAX_DEPTH + 1):
        nxt: list[tuple[list[str], Snapshot, list[str], int]] = []
        for seq, snap, children, _ in level:
            for child in children:
                g = snap.resume()
                g.step(child)
                g.generatePossibleActions()
                nodes += 1
                objs = object_list(g, name)
                key = json.dumps(normalise(objs), sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                s_ok, f_ok = match(objs)
                if s_ok and depth_struct is None:
                    depth_struct = depth
                    restore = seq + [child]
                if f_ok:
                    return depth, depth_struct, seq + [child], nodes, truncated
                if depth < MAX_DEPTH:
                    grand, cut = candidates(g, changed_names(objs, before), verb_of(child))
                    truncated = truncated or cut
                    moved, cont = changed_names(objs, before)
                    nxt.append((seq + [child], Snapshot(module, g), grand, len(moved) + len(cont)))
        if depth + 1 == MAX_DEPTH and len(nxt) > M_EXPAND:
            nxt.sort(key=lambda item: (item[3], item[0]))
            nxt, truncated = nxt[:M_EXPAND], True
        level = nxt
    return depth_full, depth_struct, restore, nodes, truncated


def label_at(depth_full: int | None, depth_struct: int | None, horizon: int) -> str:
    if depth_full is not None and depth_full <= horizon:
        return "reversible"
    if depth_struct is not None and depth_struct <= horizon:
        return "partially_reversible"
    return "irreversible"


def changed_objects(current: list[StateObject], nxt: list[StateObject]) -> list[StateObject]:
    before = {(o["name"], o["type"]): o for o in current}
    return [o for o in nxt if before.get((o["name"], o["type"])) != o]


def label_game(args: tuple[str, list[Record]]) -> tuple[str, list[Transition], dict[str, int]]:
    name, records = args
    try:
        return label_game_records(name, records)
    except Exception as exc:  # one broken game is reported in the totals and does not stop the others
        return name, [], {f"error {type(exc).__name__}: {exc}": len(records)}


def label_game_records(name: str, records: list[Record]) -> tuple[str, list[Transition], dict[str, int]]:
    module = load_game(name)
    module.UUID = 0
    game: TextGame = module.TextGame(randomSeed=module.randomSeed)
    game.generatePossibleActions()
    prefixes: dict[str, tuple[int, Snapshot]] = {}
    actions = gold_actions(name)
    for i in range(len(actions) + 1):
        key = json.dumps(normalise(object_list(game, name)), sort_keys=True)
        prefixes.setdefault(key, (i, Snapshot(module, game)))
        if i < len(actions):
            game.step(actions[i])
            game.generatePossibleActions()
    out: list[Transition] = []
    stats: Counter[str] = Counter()
    for rec in records:
        objs: list[StateObject] = rec["current_state"]["objects"]
        key = json.dumps(normalise(objs), sort_keys=True)
        if key not in prefixes:
            stats["unmatched_prefix"] += 1
            continue
        prefix_len, snap = prefixes[key]
        action = str(rec["tick_state"]["lastAction"])
        result = label_transition(module, snap, action, name)
        if result is None:
            stats["no_structural_change"] += 1
            continue
        depth_full, depth_struct, restore, nodes, truncated = result
        by_h = {str(h): label_at(depth_full, depth_struct, h) for h in (1, 2, 3)}
        nxt: list[StateObject] = rec["tick_state"]["objects"]
        out.append(Transition(
            id=f"bs32-{name}-{rec['state_id']}", game=name, state_id=int(rec["state_id"]), prefix_len=prefix_len,
            action=action, task=str(rec["current_state"]["taskDesc"]),
            current_objects=json.dumps([{k: v for k, v in o.items() if k != "uuid"} for o in objs]),
            gold_change=json.dumps([{k: v for k, v in o.items() if k != "uuid"} for o in changed_objects(objs, nxt)]),
            depth_full=depth_full, depth_struct=depth_struct, restore=restore, nodes=nodes, truncated=truncated,
            labels_by_horizon=by_h, label=by_h["3"]))
        stats["labelled"] += 1
    return name, out, dict(stats)


def cmd_label(out_path: Path, workers: int, limit_games: int) -> None:
    records: dict[str, list[Record]] = {}
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines():
        rec: Record = json.loads(line)
        records.setdefault(rec["game"], []).append(rec)
    jobs = sorted(records.items())[: limit_games or None]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    sys.setrecursionlimit(10000)
    with get_context("spawn").Pool(workers) as pool, out_path.open("w", encoding="utf-8") as f:
        for name, transitions, stats in pool.imap_unordered(label_game, jobs):
            for t in transitions:
                f.write(json.dumps(asdict(t)) + "\n")
                labels[t.label] += 1
            totals.update(stats)
            print(f"{name}: {stats}", flush=True)
    print(f"totals {dict(totals)} labels {dict(labels)}")


SYSTEM_PROMPT = ("You simulate a text-based game. Given the current game state as a JSON list of objects "
                 "and the player's action, predict the state after the action and one world tick.")
MAX_NEW_TOKENS = 640


def rq5_messages(row: dict[str, JsonValue]) -> list[dict[str, str]]:
    user = (f"Task: {row['task']}\nCurrent state:\n{row['current_objects']}\nAction: {row['action']}\n\n"
            "Output only a JSON list of the objects whose properties or contents change, in the same format "
            "as above. Output [] if nothing changes.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def canonical_change(objs: list[dict[str, JsonValue]]) -> list[str]:
    out = []
    for obj in objs:
        obj = dict(obj)
        if isinstance(obj.get("contains"), list):
            obj["contains"] = sorted(str(c) for c in obj["contains"])
        out.append(json.dumps(obj, sort_keys=True))
    return sorted(out)


def same_change(predicted: str, gold: str) -> bool:
    """Exact match of the predicted change list, ignoring object order and contents order."""
    try:
        value = json.loads(predicted[predicted.find("["):predicted.rfind("]") + 1])
    except json.JSONDecodeError:
        return False
    if not isinstance(value, list) or not all(isinstance(o, dict) for o in value):
        return False
    gold_value: list[dict[str, JsonValue]] = json.loads(gold)
    return canonical_change(value) == canonical_change(gold_value)


def cmd_nll(model_name: str, in_path: Path, out_path: Path, max_tokens: int, max_batch: int, limit: int) -> None:
    """Greedy prediction NLL (primary) and gold-change NLL (secondary) per transition."""
    import torch

    from irrev.models import device, length_batches, load, render, run_metadata

    rows = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line]
    rows = rows[:limit] if limit else rows
    done: set[str] = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}
    rows = [r for r in rows if r["id"] not in done]
    if not rows:
        print("rq5 nll: nothing to do")
        return
    model, tok = load(model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".meta.json").write_text(json.dumps(run_metadata(model_name) | {
        "max_new_tokens": str(MAX_NEW_TOKENS), "system_prompt": SYSTEM_PROMPT}, indent=1))
    prompts = [render(tok, rq5_messages(r)) for r in rows]
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) + MAX_NEW_TOKENS for p in prompts]
    configured = model.generation_config.eos_token_id
    eos_ids = {int(configured)} if isinstance(configured, int) else {int(i) for i in (configured or [])}
    eos_ids.add(int(tok.eos_token_id))

    def predict(idx: list[int]) -> list[tuple[str, float, int]]:
        try:
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device())
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None,
                                     top_k=None, pad_token_id=tok.pad_token_id, output_logits=True,
                                     return_dict_in_generate=True)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(idx) // 2
            return predict(idx[:half]) + predict(idx[half:])
        new = out.sequences[:, enc["input_ids"].shape[1]:]
        logprobs = torch.stack([torch.log_softmax(step.float(), dim=-1).gather(1, new[:, t:t + 1]).squeeze(1)
                                for t, step in enumerate(out.logits)], dim=1)
        results = []
        for b in range(len(idx)):
            tokens = new[b].tolist()
            end = next((t + 1 for t, x in enumerate(tokens) if x in eos_ids), len(tokens))
            results.append((tok.decode(tokens[:end], skip_special_tokens=True, clean_up_tokenization_spaces=False), float(-logprobs[b, :end].mean()), end))
        return results

    def gold_nll(i: int) -> float:
        enc = tok([prompts[i] + rows[i]["gold_change"]], return_tensors="pt", add_special_tokens=False,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0]
        pos = ((offsets[:, 1] > len(prompts[i])) & (offsets[:, 1] > offsets[:, 0])).nonzero().squeeze(1)
        enc = enc.to(device())
        with torch.no_grad():
            hidden = model.model(**enc).last_hidden_state[0]
            logits = model.lm_head(hidden[pos.to(device()) - 1]).float()
        target = enc["input_ids"][0, pos.to(device())]
        return float(-torch.log_softmax(logits, -1).gather(1, target.unsqueeze(1)).mean())

    start = time.time()
    batches = length_batches(lengths, max_tokens, max_batch)
    with out_path.open("a", encoding="utf-8") as f:
        for n, idx in enumerate(batches, 1):
            for i, (text, nll, n_tokens) in zip(idx, predict(idx)):
                f.write(json.dumps({"id": rows[i]["id"], "game": rows[i]["game"], "label": rows[i]["label"],
                                    "labels_by_horizon": rows[i]["labels_by_horizon"], "prompt": prompts[i],
                                    "prediction": text, "prediction_nll": nll, "prediction_tokens": n_tokens,
                                    "gold_nll": gold_nll(i), "exact_match": same_change(text, rows[i]["gold_change"])}) + "\n")
            f.flush()
            print(f"rq5 batch {n}/{len(batches)} {time.time() - start:.0f}s", flush=True)


def cmd_analyze(nll_path: Path, out_path: Path) -> None:
    """AUROC of uncertainty for irreversible vs reversible transitions, overall and within game."""
    import numpy as np
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    from irrev.stats import bootstrap_ci

    rows = [json.loads(line) for line in nll_path.read_text(encoding="utf-8").splitlines() if line]
    report: dict[str, JsonValue] = {"n": len(rows), "exact_match_rate": float(np.mean([r["exact_match"] for r in rows]))}
    for horizon in ("1", "2", "3"):
        sub = [r for r in rows if r["labels_by_horizon"][horizon] in ("reversible", "irreversible")]
        y = np.array([r["labels_by_horizon"][horizon] == "irreversible" for r in sub], dtype=int)
        if len(set(y.tolist())) < 2:
            continue
        for measure in ("prediction_nll", "gold_nll"):
            x = np.array([r[measure] for r in sub])

            def auc(i: np.ndarray) -> float:
                return float(roc_auc_score(y[i], x[i])) if len(set(y[i].tolist())) == 2 else float("nan")

            low, high = bootstrap_ci(auc, len(y))
            per_game = []
            for game in sorted({r["game"] for r in sub}):
                in_game = np.array([r["game"] == game for r in sub])
                if len(set(y[in_game].tolist())) == 2:
                    per_game.append(float(roc_auc_score(y[in_game], x[in_game])))
            report[f"h{horizon}_{measure}"] = {
                "auroc": float(roc_auc_score(y, x)), "ci_low": low, "ci_high": high,
                "n_irreversible": int(y.sum()), "n_reversible": int((1 - y).sum()),
                "mean_irreversible": float(x[y == 1].mean()), "mean_reversible": float(x[y == 0].mean()),
                "mann_whitney_p": float(mannwhitneyu(x[y == 1], x[y == 0]).pvalue),
                "within_game_auroc_mean": float(np.mean(per_game)) if per_game else float("nan"),
                "games_with_both_classes": len(per_game)}
    partial = [r["prediction_nll"] for r in rows if r["label"] == "partially_reversible"]
    report["partially_reversible_prediction_nll_mean"] = float(np.mean(partial)) if partial else float("nan")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ5 on ByteSized32-State-Prediction.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_label = sub.add_parser("label")
    p_label.add_argument("--out", type=Path, default=Path("data/rq5/transitions.jsonl"))
    p_label.add_argument("--workers", type=int, default=4)
    p_label.add_argument("--limit-games", type=int, default=0)
    p_nll = sub.add_parser("nll")
    p_nll.add_argument("--model", required=True)
    p_nll.add_argument("--transitions", type=Path, default=Path("data/rq5/transitions.jsonl"))
    p_nll.add_argument("--out", type=Path, required=True)
    p_nll.add_argument("--max-tokens", type=int, default=16000)
    p_nll.add_argument("--max-batch", type=int, default=16)
    p_nll.add_argument("--limit", type=int, default=0)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--nll", type=Path, required=True)
    p_an.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "label":
        start = time.time()
        cmd_label(args.out, args.workers, args.limit_games)
        print(f"{time.time() - start:.0f}s")
    elif args.command == "nll":
        cmd_nll(args.model, args.transitions, args.out, args.max_tokens, args.max_batch, args.limit)
    else:
        cmd_analyze(args.nll, args.out)


if __name__ == "__main__":
    main()
