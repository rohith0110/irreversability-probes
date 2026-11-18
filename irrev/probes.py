"""RQ1 probes with every compulsory control, the RQ2 comparison, and the steering direction.

Analysis plan fixed before looking at any test result (see docs/analysis_plan.md):
* Splits: by scenario, per environment 20% of scenarios for test, seeds 0, 1, 2.
  A validation split (20% of the remaining scenarios) is used only for layer selection.
* Probe: standardised features + L2 logistic regression (C=1, balanced class weights).
* Binary task = reversible vs irreversible (partial rows dropped); three-class = all rows.
* Controls: majority class, bag-of-words lexical baseline on the action text, shuffled
  labels, Hewitt-Liang control task (random fixed label per action verb, selectivity =
  probe minus control balanced accuracy), and layer 0 (embeddings).
* Best layer for steering: highest validation binary balanced accuracy on sandbox data,
  averaged over the 3 seeds, among decoder outputs 1..L-1 (the last entry is post-norm).
  Direction = mean irreversible minus mean reversible last-token state on the seed-0
  training split at that layer.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from irrev import LABEL_TO_ID
from irrev.dataset import Pair, read_pairs
from irrev.stats import bootstrap_ci

SEEDS = (0, 1, 2)
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2
C_VALUE = 1.0
Setup = Literal["binary", "three"]
SETUPS: tuple[Setup, ...] = ("binary", "three")
Position = Literal["last", "mean"]
Env = Literal["sandbox", "scienceworld", "all"]
Kind = Literal["probe", "shuffled", "control"]
Metrics = dict[str, float]


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class Data:
    rows: list[Pair]
    states: dict[str, np.ndarray]

    @property
    def n_layers(self) -> int:
        return int(self.states["last"].shape[1])


def load_data(pairs_path: Path, hidden_dir: Path) -> Data:
    rows = {r["id"]: r for r in read_pairs(pairs_path)}
    ids: list[str] = json.loads((hidden_dir / "ids.json").read_text())
    done = np.load(hidden_dir / "done.npy")
    if not done.all():
        raise RuntimeError(f"{hidden_dir}: only {int(done.sum())}/{len(done)} rows extracted")
    states = {pos: np.load(hidden_dir / f"{pos}.npy", mmap_mode="r") for pos in ("last", "mean")}
    return Data([rows[i] for i in ids], states)


def scenario_split(rows: list[Pair], seed: int) -> Split:
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for env in sorted({r["environment"] for r in rows}):
        scenarios = sorted({r["scenario_id"] for r in rows if r["environment"] == env})
        random.Random(f"{seed}:{env}").shuffle(scenarios)
        n_test = max(1, round(TEST_FRACTION * len(scenarios)))
        rest = scenarios[n_test:]
        n_val = max(1, round(VAL_FRACTION * len(rest)))
        test_s, val_s = set(scenarios[:n_test]), set(rest[:n_val])
        for i, r in enumerate(rows):
            if r["environment"] != env:
                continue
            (test if r["scenario_id"] in test_s else val if r["scenario_id"] in val_s else train).append(i)
    return Split(np.array(train), np.array(val), np.array(test))


def labels_of(rows: list[Pair], idx: np.ndarray, setup: Setup, horizon: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Keep rows usable for ``setup``; return (kept indices, integer labels)."""
    keep, ys = [], []
    for i in idx:
        label = rows[i]["labels_by_horizon"][str(horizon)] if horizon else rows[i]["label"]
        if setup == "binary" and label == "partially_reversible":
            continue
        keep.append(i)
        ys.append(int(label == "irreversible") if setup == "binary" else LABEL_TO_ID[label])
    return np.array(keep, dtype=int), np.array(ys, dtype=int)


def control_labels(rows: list[Pair], setup: Setup, seed: int) -> dict[str, int]:
    """Hewitt-Liang control task: one random fixed label per action verb type."""
    verbs = sorted({r["verb"] for r in rows})
    rng = random.Random(f"control:{seed}")
    n_classes = 2 if setup == "binary" else 3
    return {v: rng.randrange(n_classes) for v in verbs}


def score(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, n_classes: int) -> Metrics:
    out = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")), "n": float(len(y))}
    try:
        if n_classes == 2:
            out["auroc"] = float(roc_auc_score(y, proba[:, 1]))
        else:
            out["auroc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes))))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C_VALUE, class_weight="balanced", max_iter=2000))
    model.fit(x_train, y_train)
    proba = np.zeros((len(x_test), n_classes))
    proba[:, model.classes_] = model.predict_proba(x_test)
    return proba.argmax(axis=1), proba


def env_mask(rows: list[Pair], idx: np.ndarray, env: Env) -> np.ndarray:
    return idx if env == "all" else np.array([i for i in idx if rows[i]["environment"] == env], dtype=int)


@dataclass(frozen=True)
class Job:
    name: str
    train_env: Env
    test_env: Env
    setup: Setup
    position: Position
    kind: Kind
    seed: int
    layer: int
    horizon: int | None = None
    minimal_only_train: bool = False


def run_job(data: Data, job: Job) -> tuple[Job, Metrics, Metrics]:
    """Fit one probe; return metrics on the test split and on its minimal-pair subset."""
    y_te, pred, proba, minimal = predict_job(data, job)
    n_classes = 2 if job.setup == "binary" else 3
    full = score(y_te, pred, proba, n_classes)
    sub = score(y_te[minimal], pred[minimal], proba[minimal], n_classes) if minimal.sum() > 1 else {}
    return job, full, sub


def predict_job(data: Data, job: Job) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one probe; return test labels, predictions, probabilities and the minimal-pair mask."""
    rows = data.rows
    split = scenario_split(rows, job.seed)
    train_idx = np.concatenate([split.train, split.val])
    train_idx = env_mask(rows, train_idx, job.train_env)
    if job.minimal_only_train:
        train_idx = np.array([i for i in train_idx if rows[i]["pair_group"] is not None], dtype=int)
    test_idx = env_mask(rows, split.test, job.test_env)
    tr, y_tr = labels_of(rows, train_idx, job.setup, job.horizon)
    te, y_te = labels_of(rows, test_idx, job.setup, job.horizon)
    n_classes = 2 if job.setup == "binary" else 3
    if job.kind == "shuffled":
        y_tr = np.random.default_rng(job.seed).permutation(y_tr)
    if job.kind == "control":
        mapping = control_labels(rows, job.setup, job.seed)
        y_tr = np.array([mapping[rows[i]["verb"]] for i in tr])
        y_te = np.array([mapping[rows[i]["verb"]] for i in te])
    states = data.states[job.position]
    x_tr = np.asarray(states[tr, job.layer], dtype=np.float32)
    x_te = np.asarray(states[te, job.layer], dtype=np.float32)
    pred, proba = fit_predict(x_tr, y_tr, x_te, n_classes)
    minimal = np.array([rows[i]["pair_group"] is not None for i in te], dtype=bool)
    return y_te, pred, proba, minimal


def confidence_intervals(data: Data, layer: int) -> dict[str, dict[str, float]]:
    """95% bootstrap CIs (1,000 resamples, seed-0 test split) at the selected layer and at layer 0."""
    out: dict[str, dict[str, float]] = {}
    for job in jobs_for(data):
        if job.seed != 0 or job.layer not in (0, layer):
            continue
        y, pred, _, minimal = predict_job(data, job)
        key = f"{job.name}|{job.train_env}->{job.test_env}|{job.setup}|{job.position}|h{job.horizon or 3}|layer{job.layer}"
        entry = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)), "n_test": float(len(y))}
        entry["ci_low"], entry["ci_high"] = bootstrap_ci(lambda i: float(balanced_accuracy_score(y[i], pred[i])), len(y))
        if minimal.sum() > 1:
            ym, pm = y[minimal], pred[minimal]
            entry["minimal_balanced_accuracy"] = float(balanced_accuracy_score(ym, pm))
            entry["minimal_n_test"] = float(len(ym))
            entry["minimal_ci_low"], entry["minimal_ci_high"] = bootstrap_ci(lambda i: float(balanced_accuracy_score(ym[i], pm[i])), len(ym))
        out[key] = entry
    for env in ("sandbox", "scienceworld", "all"):
        for setup in ("binary", "three"):
            y, pred, _, minimal, majority = lexical_predictions(data.rows, env, setup, 0)
            for name, guess, mask in (("lexical", pred, np.ones(len(y), dtype=bool)), ("lexical_minimal", pred, minimal),
                                      ("majority", majority, np.ones(len(y), dtype=bool))):
                yy, gg = y[mask], guess[mask]
                low, high = bootstrap_ci(lambda i: float(balanced_accuracy_score(yy[i], gg[i])), len(yy))
                out[f"{name}|{env}|{setup}"] = {"balanced_accuracy": float(balanced_accuracy_score(yy, gg)),
                                                "n_test": float(len(yy)), "ci_low": low, "ci_high": high}
    return out


def lexical_baseline(rows: list[Pair], env: Env, setup: Setup, seed: int) -> tuple[Metrics, Metrics, Metrics]:
    """Bag-of-words on the action text only; returns (test, minimal-pair test, majority)."""
    y_te, pred, proba, minimal, majority = lexical_predictions(rows, env, setup, seed)
    n_classes = 2 if setup == "binary" else 3
    return (score(y_te, pred, proba, n_classes),
            score(y_te[minimal], pred[minimal], proba[minimal], n_classes) if minimal.sum() > 1 else {},
            score(y_te, majority, np.eye(n_classes)[majority], n_classes))


def lexical_predictions(rows: list[Pair], env: Env, setup: Setup, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lexical-baseline test labels, predictions, probabilities, minimal-pair mask and majority-class predictions."""
    split = scenario_split(rows, seed)
    tr, y_tr = labels_of(rows, env_mask(rows, np.concatenate([split.train, split.val]), env), setup)
    te, y_te = labels_of(rows, env_mask(rows, split.test, env), setup)
    n_classes = 2 if setup == "binary" else 3
    vectorizer = CountVectorizer(token_pattern=r"[A-Za-z_]+|\d+")
    x_tr = vectorizer.fit_transform([rows[i]["action_text"] for i in tr])
    x_te = vectorizer.transform([rows[i]["action_text"] for i in te])
    model = LogisticRegression(C=C_VALUE, class_weight="balanced", max_iter=2000).fit(x_tr, y_tr)
    proba = np.zeros((len(te), n_classes))
    proba[:, model.classes_] = model.predict_proba(x_te)
    pred = proba.argmax(axis=1)
    minimal = np.array([rows[i]["pair_group"] is not None for i in te], dtype=bool)
    majority = np.full(len(y_te), np.bincount(y_tr, minlength=n_classes).argmax())
    return y_te, pred, proba, minimal, majority


def jobs_for(data: Data) -> list[Job]:
    envs: list[Env] = [e for e in ("sandbox", "scienceworld") if any(r["environment"] == e for r in data.rows)]
    jobs: list[Job] = []
    for seed in SEEDS:
        for layer in range(data.n_layers):
            for env in envs + (["all"] if len(envs) > 1 else []):
                for setup in SETUPS:
                    jobs.append(Job("main", env, env, setup, "last", "probe", seed, layer))
                    if env != "all":
                        jobs.append(Job("token_mean", env, env, setup, "mean", "probe", seed, layer))
                        jobs.append(Job("shuffled", env, env, setup, "last", "shuffled", seed, layer))
                        jobs.append(Job("control", env, env, setup, "last", "control", seed, layer))
                        jobs.append(Job("minimal_only", env, env, setup, "last", "probe", seed, layer, minimal_only_train=True))
            if len(envs) > 1:
                for setup in SETUPS:
                    jobs.append(Job("transfer", "sandbox", "scienceworld", setup, "last", "probe", seed, layer))
                    jobs.append(Job("transfer", "scienceworld", "sandbox", setup, "last", "probe", seed, layer))
            if "scienceworld" in envs:
                for horizon in (1, 2):  # horizon 3 is the main run
                    for setup in SETUPS:
                        jobs.append(Job("horizon", "scienceworld", "scienceworld", setup, "last", "probe", seed, layer, horizon=horizon))
    return jobs


def aggregate(results: list[tuple[Job, Metrics, Metrics]]) -> dict[str, dict[str, list[float]]]:
    """Curves keyed by experiment; each metric has per-layer mean and std over seeds."""
    grouped: dict[str, dict[int, list[tuple[Metrics, Metrics]]]] = {}
    for job, full, sub in results:
        key = f"{job.name}|{job.train_env}->{job.test_env}|{job.setup}|{job.position}|h{job.horizon or 3}"
        grouped.setdefault(key, {}).setdefault(job.layer, []).append((full, sub))
    curves: dict[str, dict[str, list[float]]] = {}
    for key, by_layer in grouped.items():
        layers = sorted(by_layer)
        curve: dict[str, list[float]] = {"layer": [float(layer) for layer in layers]}
        for subset, pick in (("", 0), ("minimal_", 1)):
            for metric in ("balanced_accuracy", "macro_f1", "auroc"):
                vals = [[pair[pick].get(metric, float("nan")) for pair in by_layer[layer]] for layer in layers]
                curve[f"{subset}{metric}_mean"] = [float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan") for v in vals]
                curve[f"{subset}{metric}_std"] = [float(np.nanstd(v)) if not np.all(np.isnan(v)) else float("nan") for v in vals]
        curve["n_test"] = [float(np.mean([pair[0]["n"] for pair in by_layer[layer]])) for layer in layers]
        curves[key] = curve
    return curves


def select_layer(data: Data) -> tuple[int, list[float]]:
    """Fixed rule: best mean validation binary balanced accuracy on sandbox, layers 1..L-2."""
    rows = data.rows
    candidates = list(range(1, data.n_layers - 1))
    scores = np.zeros((len(SEEDS), len(candidates)))
    for s, seed in enumerate(SEEDS):
        split = scenario_split(rows, seed)
        tr, y_tr = labels_of(rows, env_mask(rows, split.train, "sandbox"), "binary")
        va, y_va = labels_of(rows, env_mask(rows, split.val, "sandbox"), "binary")
        states = data.states["last"]
        for c, layer in enumerate(candidates):
            pred, _ = fit_predict(np.asarray(states[tr, layer], np.float32), y_tr, np.asarray(states[va, layer], np.float32), 2)
            scores[s, c] = balanced_accuracy_score(y_va, pred)
    mean = scores.mean(axis=0)
    return candidates[int(mean.argmax())], [float(v) for v in mean]


def steering_direction(data: Data, layer: int, out_dir: Path) -> None:
    rows = data.rows
    split = scenario_split(rows, 0)
    tr, y = labels_of(rows, env_mask(rows, np.concatenate([split.train, split.val]), "sandbox"), "binary")
    x = np.asarray(data.states["last"][tr, layer], dtype=np.float32)
    direction = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
    rng = np.random.default_rng(0)
    random_dir = rng.standard_normal(direction.shape).astype(np.float32)
    random_dir *= np.linalg.norm(direction) / np.linalg.norm(random_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "direction.npy", direction.astype(np.float32))
    np.save(out_dir / "random_direction.npy", random_dir)
    return None


def stated_comparison(data: Data, stated_path: Path, layer: int) -> dict[str, float]:
    """RQ2: stated-answer accuracy and Cohen's kappa between stated answers and the probe."""
    stated = {}
    for line in stated_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "pair":
            stated[rec["id"]] = rec["stated"]
    rows = data.rows
    out: dict[str, float] = {}
    for env in ("sandbox", "scienceworld", "all"):
        accs, kappas, kappas_true, unparsed, ns = [], [], [], [], []
        for seed in SEEDS:
            split = scenario_split(rows, seed)
            tr, y_tr = labels_of(rows, env_mask(rows, np.concatenate([split.train, split.val]), env), "three")
            te, y_te = labels_of(rows, env_mask(rows, split.test, env), "three")
            if len(te) == 0:
                continue
            pred, _ = fit_predict(np.asarray(data.states["last"][tr, layer], np.float32), y_tr,
                                  np.asarray(data.states["last"][te, layer], np.float32), 3)
            said = np.array([LABEL_TO_ID.get(stated.get(rows[i]["id"], "unparsed"), -1) for i in te])
            accs.append(float((said == y_te).mean()))
            kappas.append(float(cohen_kappa_score(said, pred)))
            kappas_true.append(float(cohen_kappa_score(said, y_te)))
            unparsed.append(float((said == -1).mean()))
            ns.append(float(len(te)))
            if seed == 0:
                hit = (said == y_te).astype(float)
                low, high = bootstrap_ci(lambda idx: float(hit[idx].mean()), len(hit))
                out[f"{env}_stated_accuracy_seed0_ci_low"], out[f"{env}_stated_accuracy_seed0_ci_high"] = low, high
        if accs:
            out.update({f"{env}_stated_accuracy_mean": float(np.mean(accs)), f"{env}_stated_accuracy_std": float(np.std(accs)),
                        f"{env}_kappa_stated_probe_mean": float(np.mean(kappas)), f"{env}_kappa_stated_probe_std": float(np.std(kappas)),
                        f"{env}_kappa_stated_truth_mean": float(np.mean(kappas_true)),
                        f"{env}_unparsed_rate": float(np.mean(unparsed)), f"{env}_n_test_mean": float(np.mean(ns))})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Probes, controls, ablations and steering direction.")
    parser.add_argument("command", choices=("all", "select-layer", "ci", "stated"))
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs.jsonl"))
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stated", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    data = load_data(args.pairs, args.hidden)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command in ("ci", "stated"):  # recompute only what depends on the selected layer
        layer = int(json.loads((args.out / "best_layer.json").read_text())["hidden_index"])
        if args.command == "ci":
            (args.out / "ci.json").write_text(json.dumps(confidence_intervals(data, layer), indent=1))
        elif args.stated is not None:
            (args.out / "stated_vs_probe.json").write_text(json.dumps(stated_comparison(data, args.stated, layer), indent=1))
        print(f"{args.command}: layer {layer}, written to {args.out}")
        return
    best_path = args.out / "best_layer.json"
    if best_path.exists():
        # The layer and steering direction are chosen once (the GPU queue's select-layer job) and never
        # recomputed: near-tied layers can flip on other hardware, and steering runs read these files.
        layer = int(json.loads(best_path.read_text())["hidden_index"])
        print(f"best layer {layer} (existing selection kept)")
    else:
        layer, val_curve = select_layer(data)
        steering_direction(data, layer, args.out)
        best_path.write_text(json.dumps({"hidden_index": layer, "validation_balanced_accuracy": val_curve,
                                         "rule": "max mean val binary balanced accuracy, sandbox, layers 1..L-2"}, indent=2))
        print(f"best layer {layer}")
    if args.command == "select-layer":
        return

    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_job)(data, job) for job in jobs_for(data))
    (args.out / "curves.json").write_text(json.dumps(aggregate(results), indent=1))
    baselines: dict[str, dict[str, Metrics]] = {}
    for env in ("sandbox", "scienceworld", "all"):
        if env != "all" and not any(r["environment"] == env for r in data.rows):
            continue
        for setup in ("binary", "three"):
            for seed in SEEDS:
                lex, lex_min, majority = lexical_baseline(data.rows, env, setup, seed)
                baselines[f"{env}|{setup}|seed{seed}"] = {"lexical": lex, "lexical_minimal": lex_min, "majority": majority}
    (args.out / "baselines.json").write_text(json.dumps(baselines, indent=1))
    if args.stated is not None:
        (args.out / "stated_vs_probe.json").write_text(json.dumps(stated_comparison(data, args.stated, layer), indent=1))


if __name__ == "__main__":
    main()
