"""Figures and tables required by brief section 8, from saved result files only.

* probe accuracy vs layer with baselines (per model, environment, task)
* stated vs probed vs behaviour comparison
* blindness gap with 95% CIs
* ablation table (markdown)
* steering curve (caution and success vs strength, probe vs random direction)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

Curve = dict[str, list[float]]
COLORS = {"main": "#1f5aa6", "token_mean": "#6a9fd8", "shuffled": "#9a9a9a", "control": "#c7792b", "minimal_only": "#3a8f5b"}


def load_json(path: Path) -> dict[str, Curve]:
    loaded: dict[str, Curve] = json.loads(path.read_text())
    return loaded


def probe_plot(curves: dict[str, Curve], baselines: dict[str, dict[str, dict[str, float]]], env: str, setup: str,
               title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name in ("main", "token_mean", "shuffled", "control", "minimal_only"):
        key = f"{name}|{env}->{env}|{setup}|{'mean' if name == 'token_mean' else 'last'}|h3"
        if key not in curves:
            continue
        c = curves[key]
        mean = c["balanced_accuracy_mean"]
        std = c["balanced_accuracy_std"]
        ax.plot(c["layer"], mean, label=name.replace("_", " "), color=COLORS[name], lw=1.8)
        ax.fill_between(c["layer"], [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                        color=COLORS[name], alpha=0.15, lw=0)
    main_key = f"main|{env}->{env}|{setup}|last|h3"
    if main_key in curves and "minimal_balanced_accuracy_mean" in curves[main_key]:
        c = curves[main_key]
        ax.plot(c["layer"], c["minimal_balanced_accuracy_mean"], ls="--", color=COLORS["main"], lw=1.2,
                label="probe on minimal pairs")
    lex = [v["lexical"]["balanced_accuracy"] for k, v in baselines.items() if k.startswith(f"{env}|{setup}|")]
    lex_min = [v["lexical_minimal"].get("balanced_accuracy", float("nan")) for k, v in baselines.items() if k.startswith(f"{env}|{setup}|")]
    maj = [v["majority"]["balanced_accuracy"] for k, v in baselines.items() if k.startswith(f"{env}|{setup}|")]
    for values, style, label in ((lex, "-.", "lexical (all)"), (lex_min, ":", "lexical (minimal pairs)"), (maj, (0, (1, 3)), "majority")):
        if values:
            ax.axhline(sum(values) / len(values), color="#444444", ls=style, lw=1, label=label)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def gap_plot(models: dict[str, dict[str, dict[str, float]]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    names, values, lows, highs = [], [], [], []
    for model, report in models.items():
        for condition in ("neutral", "careful"):
            gap = report.get(condition, {}).get("blindness_gap")
            if isinstance(gap, dict):
                names.append(f"{model}\n{condition}")
                values.append(gap["value"])
                lows.append(gap["value"] - gap["ci_low"])
                highs.append(gap["ci_high"] - gap["value"])
    ax.bar(range(len(values)), values, yerr=[lows, highs], color="#1f5aa6", capsize=4)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(range(len(values)), names, fontsize=8)
    ax.set_ylabel("blindness gap K - A (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def comparison_plot(models: dict[str, tuple[dict[str, float], dict[str, Curve], dict[str, dict[str, dict[str, float]]], int]], out: Path) -> None:
    """Stated accuracy vs probe balanced accuracy at the selected layer vs behavioural sensitivity."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    width = 0.25
    for m, (model, (stated, curves, behaviour, layer)) in enumerate(models.items()):
        probe = curves.get("main|sandbox->sandbox|three|last|h3", {}).get("balanced_accuracy_mean", [])
        sens = behaviour.get("neutral", {}).get("reversibility_sensitivity", {})
        bars = [stated.get("sandbox_stated_accuracy_mean", float("nan")),
                probe[layer] if layer < len(probe) else float("nan"),
                sens.get("value", float("nan")) if isinstance(sens, dict) else float("nan")]
        ax.bar([i + m * width for i in range(3)], bars, width, label=model)
    ax.set_xticks([i + width / 2 for i in range(3)], ["stated accuracy", "probe bal. acc.\n(selected layer)", "behaviour\nsensitivity"], fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def steering_plot(report: dict[str, dict[str, dict[str, float]]], out: Path) -> None:
    pattern = re.compile(r"steer_(direction|random_direction)_([+-]?\d+(?:\.\d+)?)")
    series: dict[str, list[tuple[float, float, float, float]]] = {}
    for name, metrics in report.items():
        match = pattern.fullmatch(name)
        if not match:
            continue
        caution = metrics["caution_rate_irreversible_version"]
        success = metrics["success_rate_irreversible_version"]
        series.setdefault(match.group(1), []).append((float(match.group(2)), caution["rate"], success["rate"], caution["ci_high"] - caution["rate"]))
    if "direction" in series and "steer_direction_+0" in report:
        series.setdefault("random_direction", []).append(
            (0.0, report["steer_direction_+0"]["caution_rate_irreversible_version"]["rate"],
             report["steer_direction_+0"]["success_rate_irreversible_version"]["rate"], 0.0))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for kind, color in (("direction", "#1f5aa6"), ("random_direction", "#9a9a9a")):
        pts = sorted(series.get(kind, []))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ax.errorbar(xs, [p[1] for p in pts], yerr=[p[3] for p in pts], marker="o", color=color, label=f"caution ({kind.replace('_', ' ')})", capsize=3)
        ax.plot(xs, [p[2] for p in pts], marker="s", ls="--", color=color, label=f"task success ({kind.replace('_', ' ')})")
    ax.set_xlabel("steering strength (multiples of the direction)")
    ax.set_ylabel("rate (irreversible versions)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def rq5_plot(nll_path: Path, model: str, out: Path) -> None:
    """Prediction NLL by reversibility class (horizon 3)."""
    rows = [json.loads(line) for line in nll_path.read_text(encoding="utf-8").splitlines() if line]
    classes = ["reversible", "partially_reversible", "irreversible"]
    data = [[r["prediction_nll"] for r in rows if r["label"] == c] for c in classes]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2, 3], [f"{c.replace('_', ' ')}\n(n={len(d)})" for c, d in zip(classes, data)], fontsize=8)
    ax.set_ylabel("mean token NLL of own prediction")
    ax.set_title(f"{model}: next-state uncertainty (RQ5)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def ablation_table(curves: dict[str, Curve], layer: int) -> str:
    rows = ["| experiment | train -> test | task | position | horizon | bal. acc. (mean ± sd) | minimal-pair bal. acc. |",
            "|---|---|---|---|---|---|---|"]
    for key in sorted(curves):
        name, envs, setup, position, horizon = key.split("|")
        c = curves[key]
        if layer >= len(c["balanced_accuracy_mean"]):
            continue
        minimal = c.get("minimal_balanced_accuracy_mean", [float("nan")] * (layer + 1))[layer]
        rows.append(f"| {name} | {envs} | {setup} | {position} | {horizon} | "
                    f"{c['balanced_accuracy_mean'][layer]:.3f} ± {c['balanced_accuracy_std'][layer]:.3f} | {minimal:.3f} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make every report figure and table from results/.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="*", default=["qwen7b", "llama8b"])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gap_inputs: dict[str, dict[str, dict[str, float]]] = {}
    comparison: dict[str, tuple[dict[str, float], dict[str, Curve], dict[str, dict[str, dict[str, float]]], int]] = {}
    for model in args.models:
        probes = args.results / model / "probes"
        if not (probes / "curves.json").exists():
            print(f"{model}: no probe results yet")
            continue
        curves = load_json(probes / "curves.json")
        baselines = json.loads((probes / "baselines.json").read_text())
        layer = int(json.loads((probes / "best_layer.json").read_text())["hidden_index"])
        for env in ("sandbox", "scienceworld", "all"):
            for setup in ("binary", "three"):
                if f"main|{env}->{env}|{setup}|last|h3" in curves:
                    probe_plot(curves, baselines, env, setup, f"{model}: {env}, {setup}", args.out / f"probe_{model}_{env}_{setup}.png")
        (args.out / f"ablations_{model}.md").write_text(ablation_table(curves, layer) + "\n", encoding="utf-8")
        behaviour_path = args.results / model / "behaviour.json"
        behaviour = json.loads(behaviour_path.read_text()) if behaviour_path.exists() else {}
        gap_inputs[model] = behaviour
        stated_path = probes / "stated_vs_probe.json"
        comparison[model] = (json.loads(stated_path.read_text()) if stated_path.exists() else {}, curves, behaviour, layer)
        rq5_path = args.results / model / "rq5" / "nll.jsonl"
        if rq5_path.exists():
            rq5_plot(rq5_path, model, args.out / f"rq5_{model}.png")
        steering_path = args.results / model / "steering_behaviour.json"
        if steering_path.exists():
            steering_plot(json.loads(steering_path.read_text()), args.out / f"steering_{model}.png")
    if gap_inputs:
        gap_plot(gap_inputs, args.out / "blindness_gap.png")
        comparison_plot(comparison, args.out / "stated_probe_behaviour.png")
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
