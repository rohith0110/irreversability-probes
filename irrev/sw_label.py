"""ScienceWorld reversibility labelling (brief section 4.2).

ScienceWorld has no save/restore, so every search node is reached by resetting the
task variation and replaying its action history.

Label definition (fixed before any model results were seen):
* The world evolves on its own every tick (temperatures drift), so a candidate state
  is compared with the *null baseline*: the states reached from the same start by 0 to
  4 no-op ticks (``look at agent``). Matching any of them counts, because failed actions
  take no time and the move counter does not track world time reliably in every task.
* reversible at horizon h: some sequence of d <= h actions reaches a state whose
  structure (object tree, containment and every non-numeric property) equals the null
  baseline and whose temperatures are all within tolerance: ``TEMP_TOL`` degrees C, or
  ``TEMP_TOL + NOISE_FACTOR * drift`` for objects whose temperature drifts between two
  replays of the same history (the simulator is not numerically reproducible).
* A restoring action that triggers the parser's "Ambiguous request" prompt is tried with
  each numbered option. A labelled action that is ambiguous is skipped, and so is one
  that leaves the structure unchanged (a failed action whose only effect is elapsed time).
* partially reversible at horizon h: not reversible, but the structure is restored
  within h actions (only continuous quantities differ: "restored approximately").
* irreversible: neither, within h actions.

Search: breadth-first over valid actions up to depth 3, restricted to restoring verbs
whose first object (by simulator object id) differs from the target, or "go to" a
container that changed (connect/disconnect only when the labelled action is itself
electrical); ranked by changed patient, then inverse verb, at most ``K``
children per node and ``M`` expanded depth-2 nodes. Any pruning sets ``truncated`` so it is reported.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import TypedDict

NOOP = "look at agent"
NUMERIC_KEYS = ("combustionTicks", "temperatureC")
PROP_KEYS = (
    "propMaterial", "propEdibility", "propContainer", "propDevice", "propHeatSource",
    "propCoolingSource", "propPortal", "propMoveable", "propElectricalConnection", "propLife",
)
TEMP_TOL = 0.5
NOISE_FACTOR = 3.0
AMBIGUITY_SEP = " ##"
MAX_DEPTH = 3
K_CHILDREN = 10
M_EXPAND = 6

VERBS = (
    "pick up", "put down", "go to", "focus on", "look around", "look at", "look in",
    "teleport to", "deactivate", "activate", "disconnect", "connect", "open", "close",
    "move", "pour", "mix", "eat", "dunk", "use", "flush", "examine", "read", "wait1",
    "wait", "inventory", "task", "reset",
)
THERMAL_WORDS = ("stove", "oven", "freezer", "fridge", "fire pit", "blast furnace", "hot plate", "lighter")
THERMAL_TASKS = (
    "boil", "melt", "freeze", "change-the-state-of-matter-of", "use-thermometer",
    "measure-melting-point-known-substance", "measure-melting-point-unknown-substance",
)
NON_SAMPLED_VERBS = {
    "focus on", "look around", "look at", "look in", "examine", "read", "inventory",
    "task", "wait", "wait1", "reset", "teleport to",
}
RESTORING_VERBS = {
    "open", "close", "activate", "deactivate", "move", "pick up", "put down", "pour",
    "go to", "connect", "disconnect", "dunk",
}
ELECTRICAL_VERBS = {"connect", "disconnect"}
INVERSE_VERBS = {
    "open": {"close"}, "close": {"open"}, "activate": {"deactivate"},
    "deactivate": {"activate"}, "pick up": {"put down", "move"},
    "put down": {"pick up", "move"}, "move": {"move", "pick up"},
    "connect": {"disconnect"}, "disconnect": {"connect"}, "go to": {"go to"},
    "pour": {"pour"}, "dunk": {"move", "pick up"},
}

PropValue = str | float | int | bool | None | list[str]


class SWObject(TypedDict):
    uuid: str
    name: str
    isDeleted: bool
    contents: dict[str, "SWObject"]
    propMaterial: dict[str, PropValue] | None
    propEdibility: dict[str, PropValue] | None
    propContainer: dict[str, PropValue] | None
    propDevice: dict[str, PropValue] | None
    propHeatSource: dict[str, PropValue] | None
    propCoolingSource: dict[str, PropValue] | None
    propPortal: dict[str, PropValue] | None
    propMoveable: dict[str, PropValue] | None
    propElectricalConnection: dict[str, PropValue] | None
    propLife: dict[str, PropValue] | None


@dataclass(frozen=True)
class Canon:
    """Canonical world state: structural multiset plus numeric values per entry."""

    struct: tuple[str, ...]
    numeric: tuple[tuple[str, tuple[float, ...]], ...]
    owners: tuple[tuple[str, str, str], ...] = field(default=(), compare=False)  # (key, uuid, parent uuid)

    def key(self) -> str:
        rounded = [(k, tuple(round(v, 1) for v in vals)) for k, vals in self.numeric]
        return hashlib.sha1(json.dumps([self.struct, rounded]).encode()).hexdigest()


@dataclass
class PairRecord:
    id: str
    environment: str
    scenario_id: str
    task: str
    variation: int
    history: list[str]
    state_text: str
    action_text: str
    verb: str
    observation: str
    depth_full: int | None
    depth_struct: int | None
    restore_full: list[str] | None
    restore_struct: list[str] | None
    nodes: int
    truncated: bool
    labels_by_horizon: dict[str, str]
    label: str
    horizon: int
    source: str
    seconds: float


def verb_of(action: str) -> str:
    for verb in VERBS:
        if action == verb or action.startswith(verb + " "):
            return verb
    return action.split(" ", 1)[0]


class ValidAction(TypedDict):
    action: str
    template_id: int
    obj_ids: list[int]
    type_ids: list[int]


def canonicalize(tree: SWObject) -> Canon:
    entries: list[tuple[str, tuple[float, ...]]] = []
    owners: list[tuple[str, str, str]] = []

    def walk(node: SWObject, parent: str, parent_uuid: str) -> None:
        path = f"{parent}/{node['name']}"
        props: dict[str, dict[str, PropValue]] = {}
        numbers: list[float] = []
        for prop_key in PROP_KEYS:
            prop = node.get(prop_key)
            if prop is None:
                continue
            props[prop_key] = {k: v for k, v in prop.items() if k not in NUMERIC_KEYS}
            for num_key in NUMERIC_KEYS:
                value = prop.get(num_key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numbers.append(float(value))
        signature = json.dumps({"deleted": node["isDeleted"], "props": props}, sort_keys=True)
        key = f"{path}|{signature}"
        entries.append((key, tuple(numbers)))
        owners.append((key, str(node["uuid"]), parent_uuid))
        for child in node["contents"].values():
            walk(child, path, str(node["uuid"]))

    walk(tree, "", "")
    entries.sort()
    return Canon(struct=tuple(k for k, _ in entries), numeric=tuple(entries), owners=tuple(owners))


def numerically_close(a: Canon, b: Canon, tolerance: dict[str, float]) -> bool:
    """Equal structure assumed; every numeric value within its entry's tolerance."""
    return all(
        len(va) == len(vb) and all(abs(x - y) <= tolerance.get(key, TEMP_TOL) for x, y in zip(va, vb))
        for (key, va), (_, vb) in zip(a.numeric, b.numeric)
    )


def replay_noise(a: Canon, b: Canon) -> dict[str, float]:
    """Per-entry tolerance from two replays of the same history (the simulator's heat
    transfer is not bit-reproducible): TEMP_TOL + NOISE_FACTOR * observed drift."""
    if a.struct != b.struct:
        return {}
    out: dict[str, float] = {}
    for (key, va), (_, vb) in zip(a.numeric, b.numeric):
        drift = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        if drift > 0:
            out[key] = max(out.get(key, TEMP_TOL), TEMP_TOL + NOISE_FACTOR * drift)
    return out


def ambiguity_options(observation: str) -> int:
    """Number of numbered choices when the parser asks which object was meant, else 0."""
    if not observation.startswith("Ambiguous request"):
        return 0
    return sum(1 for line in observation.splitlines() if line.split(":", 1)[0].strip().isdigit())


def changed_objects(current: Canon, target: Canon) -> tuple[set[str], set[str], set[str]]:
    """(directly changed, changed incl. inherited path changes, their containers) as UUIDs.

    Directly changed: created or deleted, moved to another container, or own properties
    changed. Objects whose path changed only because an ancestor moved are not direct.
    """
    diff = Counter(current.struct)
    diff.subtract(Counter(target.struct))
    keys = {k for k, c in diff.items() if c}
    cur = {uuid: (key, parent) for key, uuid, parent in current.owners}
    tgt = {uuid: (key, parent) for key, uuid, parent in target.owners}
    moved: set[str] = set()
    containers: set[str] = set()
    direct: set[str] = set()
    for uuid in set(cur) | set(tgt):
        a, b = cur.get(uuid), tgt.get(uuid)
        if (a and a[0] in keys) or (b and b[0] in keys):
            moved.add(uuid)
            containers.update(x[1] for x in (a, b) if x)
            if a is None or b is None or a[1] != b[1]:
                direct.add(uuid)
            elif a[0].split("|", 1)[1] != b[0].split("|", 1)[1]:
                direct.add(uuid)
    return direct, moved, containers


def struct_distance(current: Canon, target: Canon) -> int:
    diff = Counter(current.struct)
    diff.subtract(Counter(target.struct))
    return sum(abs(c) for c in diff.values())


def restore_candidates(valid: list[ValidAction], current: Canon, target: Canon, last_verb: str,
                       electrical: bool) -> tuple[list[str], bool]:
    """Restoring-verb actions whose patient (first object) changed, or that go to a changed container."""
    direct, moved, containers = changed_objects(current, target)
    inverse = INVERSE_VERBS.get(last_verb, set())
    scored: list[tuple[int, int, int, str]] = []
    for item in valid:
        action, ids = item["action"], [str(i) for i in item["obj_ids"]]
        verb = verb_of(action)
        if not ids or verb not in RESTORING_VERBS or (verb in ELECTRICAL_VERBS and not electrical):
            continue
        patient_changed = ids[0] in moved
        if not (patient_changed or (verb == "go to" and ids[0] in containers)):
            continue
        hits = sum(1 for i in ids if i in moved or i in containers)
        rank = 0 if ids[0] in direct else 1 if patient_changed else 2
        scored.append((rank, 0 if verb in inverse else 1, -hits, action))
    scored.sort()
    return [item[-1] for item in scored[:K_CHILDREN]], len(scored) > K_CHILDREN


class Replayer:
    """One JVM-backed ScienceWorld instance that reaches states by reset + replay."""

    def __init__(self) -> None:
        from scienceworld import ScienceWorldEnv

        self.env = ScienceWorldEnv("", envStepLimit=1_000_000)
        self.loaded: tuple[str, int] | None = None
        self.gold: list[str] = []

    def load(self, task: str, variation: int) -> list[str]:
        if self.loaded != (task, variation):
            self.env.load(task, variation, "", True)
            self.gold = list(self.env.get_gold_action_sequence())
            self.loaded = (task, variation)
        return self.gold

    def replay(self, actions: list[str]) -> str:
        """Reset and replay; ``"<action> ##k"`` answers an ambiguity prompt with option k."""
        self.env.server.reset()
        observation = ""
        for action in actions:
            base, _, choice = action.partition(AMBIGUITY_SEP)
            observation = self.env.server.step(base)
            if choice and ambiguity_options(observation):
                observation = self.env.server.step(choice)
        return observation

    def ended(self) -> bool:
        return bool(self.env.server.getCompleted()) or self.env.server.getScore() < 0

    def canon(self) -> Canon:
        return canonicalize(self.env.getObjectTree())

    def valid_actions(self) -> list[ValidAction]:
        actions: list[ValidAction] = self.env.get_valid_action_object_combinations_with_templates()
        return actions

    def close(self) -> None:
        self.env.close()


@dataclass
class SearchResult:
    observation: str
    depth_full: int | None
    depth_struct: int | None
    restore_full: list[str] | None
    restore_struct: list[str] | None
    nodes: int
    truncated: bool


def search(rep: Replayer, history: list[str], action: str) -> SearchResult | None:
    """Return the minimal restoring depths for ``action`` at the state after ``history``.

    A node is restored if it matches the null baseline at any tick 0..MAX_DEPTH+1: the
    simulator's move counter does not track world time reliably in every task.
    """
    def null_run() -> list[Canon]:
        rep.replay(history)
        states = [rep.canon()]
        for _ in range(MAX_DEPTH + 1):
            rep.env.server.step(NOOP)
            states.append(rep.canon())
        return states

    nulls = null_run()
    again = null_run()
    tolerance = replay_noise(nulls[0], again[0])
    for key, tol in replay_noise(nulls[-1], again[-1]).items():
        tolerance[key] = max(tol, tolerance.get(key, TEMP_TOL))

    def match(state: Canon) -> tuple[bool, bool]:
        same = [n for n in nulls if n.struct == state.struct]
        return bool(same), any(numerically_close(state, n, tolerance) for n in same)

    observation = rep.replay(history + [action])
    if rep.ended() or ambiguity_options(observation):
        return None
    root = rep.canon()
    struct_ok, _ = match(root)
    if struct_ok:  # no structural effect: failed or time-only actions are not labelled
        return SearchResult(observation, 0, 0, [], [], 0, False)

    depth_struct: int | None = None
    restore_struct: list[str] | None = None
    electrical = verb_of(action) in ELECTRICAL_VERBS
    cands, truncated = restore_candidates(rep.valid_actions(), root, nulls[0], verb_of(action), electrical)
    level: list[tuple[list[str], list[str], int]] = [([], cands, 0)]
    seen = {root.key()}
    nodes = 0

    for depth in range(1, MAX_DEPTH + 1):
        next_level: list[tuple[list[str], list[str], int]] = []
        for seq, children, _ in level:
            queue = list(children)
            while queue:
                child = queue.pop(0)
                new_seq = seq + [child]
                observation_child = rep.replay(history + [action] + new_seq)
                nodes += 1
                options = ambiguity_options(observation_child)
                if options and AMBIGUITY_SEP not in child:
                    queue[:0] = [f"{child}{AMBIGUITY_SEP}{k}" for k in range(options)]
                    continue
                if rep.ended():
                    continue
                state = rep.canon()
                key = state.key()
                if key in seen:
                    continue
                seen.add(key)
                struct_ok, full_ok = match(state)
                if struct_ok and depth_struct is None:
                    depth_struct, restore_struct = depth, new_seq
                if full_ok:
                    return SearchResult(observation, depth, depth_struct, new_seq, restore_struct, nodes, truncated)
                if depth < MAX_DEPTH:
                    last_verb = verb_of(child.partition(AMBIGUITY_SEP)[0])
                    grand, cut = restore_candidates(rep.valid_actions(), state, nulls[0], last_verb, electrical)
                    truncated = truncated or cut
                    next_level.append((new_seq, grand, struct_distance(state, nulls[0])))
        if depth + 1 == MAX_DEPTH and len(next_level) > M_EXPAND:
            next_level.sort(key=lambda item: (item[2], item[0]))
            next_level = next_level[:M_EXPAND]
            truncated = True
        level = next_level

    return SearchResult(observation, None, depth_struct, None, restore_struct, nodes, truncated)


def label_at(result: SearchResult, horizon: int) -> str:
    if result.depth_full is not None and result.depth_full <= horizon:
        return "reversible"
    if result.depth_struct is not None and result.depth_struct <= horizon:
        return "partially_reversible"
    return "irreversible"


def sample_actions(valid: list[str], gold_next: str | None, rng: random.Random, n: int, thermal: bool = False) -> list[str]:
    by_verb: dict[str, list[str]] = {}
    for action in valid:
        verb = verb_of(action)
        if verb not in NON_SAMPLED_VERBS:
            by_verb.setdefault(verb, []).append(action)
    picked: list[str] = []
    if gold_next and verb_of(gold_next) not in NON_SAMPLED_VERBS and gold_next in valid:
        picked.append(gold_next)
    if thermal:
        heat = sorted(a for a in valid if verb_of(a) in ("move", "pour", "dunk", "activate", "deactivate", "pick up")
                      and any(w in a for w in THERMAL_WORDS))
        picked += [a for a in rng.sample(heat, min(2, len(heat))) if a not in picked]
    verbs = sorted(by_verb)
    rng.shuffle(verbs)
    for verb in verbs:
        if len(picked) >= n:
            break
        choice = rng.choice(sorted(by_verb[verb]))
        if choice not in picked:
            picked.append(choice)
    return picked


def state_text(rep: Replayer) -> str:
    return f"Task: {rep.env.get_task_description()}\n{rep.env.look().strip()}\n{rep.env.inventory().strip()}"


_REPLAYER: Replayer | None = None


def _worker_init() -> None:
    global _REPLAYER
    _REPLAYER = Replayer()


def label_scenario(job: tuple[str, int, int, int, int, int, bool]) -> tuple[str, list[PairRecord], str]:
    task, variation, seed, n_prefixes, n_actions, max_prefix, thermal = job
    scenario_id = f"sw::{task}::{variation}"
    rep = _REPLAYER
    if rep is None:
        raise RuntimeError("worker not initialised")
    rng = random.Random(f"{seed}:{scenario_id}")
    records: list[PairRecord] = []
    try:
        gold = rep.load(task, variation)
        usable = [a for a in gold if not a.startswith("ERROR")]
        limit = min(len(usable) - 1, max_prefix)
        if limit < 1:
            return scenario_id, records, "no gold path"
        low = limit // 2 if thermal else 0  # thermal pass: later states, when heat sources are in use
        prefixes = sorted(rng.sample(range(low, limit + 1), min(n_prefixes, limit + 1 - low)))
        for plen in prefixes:
            history = usable[:plen]
            rep.replay(history)
            if rep.ended():
                continue
            text = state_text(rep)
            valid = [a["action"] for a in rep.valid_actions()]
            gold_next = usable[plen] if plen < len(usable) else None
            for action in sample_actions(valid, gold_next, rng, n_actions, thermal):
                start = time.time()
                result = search(rep, history, action)
                if result is None or result.depth_full == 0:
                    continue
                by_h = {str(h): label_at(result, h) for h in (1, 2, 3)}
                pair_id = hashlib.sha1(f"{scenario_id}|{plen}|{action}".encode()).hexdigest()[:12]
                record = PairRecord(
                    id=f"sw-{pair_id}", environment="scienceworld", scenario_id=scenario_id,
                    task=task, variation=variation, history=history, state_text=text,
                    action_text=action, verb=verb_of(action), observation=result.observation,
                    depth_full=result.depth_full, depth_struct=result.depth_struct,
                    restore_full=result.restore_full, restore_struct=result.restore_struct,
                    nodes=result.nodes, truncated=result.truncated, labels_by_horizon=by_h,
                    label=by_h["3"], horizon=3, source="scienceworld_bfs_replay",
                    seconds=round(time.time() - start, 2),
                )
                records.append(record)
        return scenario_id, records, "ok"
    except Exception as exc:  # one broken scenario is logged and must not stop the overnight run
        return scenario_id, records, f"error: {type(exc).__name__}: {exc}"


def build_jobs(seed: int, variations_per_task: int, n_prefixes: int, n_actions: int, max_prefix: int,
               tasks: list[str] | None, thermal: bool) -> list[tuple[str, int, int, int, int, int, bool]]:
    rep = Replayer()
    try:
        names = tasks or rep.env.get_task_names()
        rng = random.Random(seed)
        jobs = []
        for task in names:
            n_var = rep.env.get_max_variations(task)
            for variation in sorted(rng.sample(range(n_var), min(variations_per_task, n_var))):
                jobs.append((task, variation, seed, n_prefixes, n_actions, max_prefix, thermal))
    finally:
        rep.close()
    rng.shuffle(jobs)
    return jobs


def lower_priority() -> None:
    if sys.platform == "win32":
        below_normal = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw/scienceworld_pairs.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variations-per-task", type=int, default=5)
    parser.add_argument("--prefixes", type=int, default=3)
    parser.add_argument("--actions", type=int, default=4)
    parser.add_argument("--max-prefix", type=int, default=25)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--thermal", action="store_true",
                        help="targeted pass: thermal tasks, later states, extra moves onto heat or cold sources")
    args = parser.parse_args()

    lower_priority()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_path = args.out.with_suffix(".done.jsonl")
    done = set()
    if done_path.exists():
        done = {json.loads(line)["scenario_id"] for line in done_path.read_text(encoding="utf-8").splitlines() if line}

    tasks = args.tasks or (list(THERMAL_TASKS) if args.thermal else None)
    jobs = build_jobs(args.seed, args.variations_per_task, args.prefixes, args.actions, args.max_prefix, tasks, args.thermal)
    jobs = [j for j in jobs if f"sw::{j[0]}::{j[1]}" not in done]
    if args.limit_jobs:
        jobs = jobs[: args.limit_jobs]
    print(f"{len(jobs)} scenarios to label, {len(done)} already done", flush=True)

    counts: Counter[str] = Counter()
    start = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init) as pool, \
            args.out.open("a", encoding="utf-8") as out, done_path.open("a", encoding="utf-8") as done_f:
        for i, (scenario_id, records, status) in enumerate(pool.imap_unordered(label_scenario, jobs), 1):
            for record in records:
                out.write(json.dumps(asdict(record)) + "\n")
                counts[record.label] += 1
            out.flush()
            done_f.write(json.dumps({"scenario_id": scenario_id, "status": status, "n": len(records)}) + "\n")
            done_f.flush()
            elapsed = time.time() - start
            print(f"[{i}/{len(jobs)}] {scenario_id} {status} +{len(records)} total={dict(counts)} {elapsed / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
