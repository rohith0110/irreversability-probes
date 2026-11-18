"""Tables, number macros and combined figures for the paper, from saved result files only.

    python -m irrev.paper            # report/generated/*.tex and report/figures/*.pdf

Every number in report/main.tex comes from a macro or table written here, so the paper is
reproduced by re-running this script after the analysis commands in the README.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results")
GENERATED = Path("report/generated")
FIGURES = Path("report/figures")
MODELS = {"qwen7b": ("Qwen2.5-7B", "QwenSeven"), "llama8b": ("Llama-3.1-8B", "LlamaEight"),
          "qwen14b": ("Qwen2.5-14B", "QwenFourteen")}
COLORS = {"qwen7b": "#1f5aa6", "llama8b": "#c7792b", "qwen14b": "#3a8f5b"}

Curves = dict[str, dict[str, list[float]]]
Json = dict[str, "JsonValue"]
JsonValue = float | int | str | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def load(path: Path) -> Json:
    value: Json = json.loads(path.read_text(encoding="utf-8"))
    return value


def num(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def sub(value: JsonValue, key: str) -> Json:
    if not isinstance(value, dict) or not isinstance(value.get(key), dict):
        raise KeyError(key)
    inner = value[key]
    assert isinstance(inner, dict)
    return inner


def f3(x: float) -> str:
    return "--" if math.isnan(x) else f"{x:.3f}".replace("-", "$-$") if x < 0 else f"{x:.3f}"


def f2(x: float) -> str:
    return "--" if math.isnan(x) else (f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}")


def pval(p: float) -> str:
    """Relation and value for math mode: '<0.001' or '=0.062'."""
    return "<0.001" if p < 0.001 else f"={p:.2g}"


def ci(entry: Json, value_key: str = "rate") -> str:
    return f"{f2(num(entry[value_key]))} [{f2(num(entry['ci_low']))}, {f2(num(entry['ci_high']))}]"


def available() -> list[str]:
    return [m for m in MODELS if (RESULTS / m / "probes" / "curves.json").exists()]


def curves_of(model: str) -> Curves:
    loaded: Curves = json.loads((RESULTS / model / "probes" / "curves.json").read_text())
    return loaded


def layer_of(model: str) -> int:
    return int(num(load(RESULTS / model / "probes" / "best_layer.json")["hidden_index"]))


def at(curves: Curves, key: str, layer: int, metric: str = "balanced_accuracy_mean") -> float:
    values = curves.get(key, {}).get(metric, [])
    return float(values[layer]) if layer < len(values) else float("nan")


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


# --------------------------------------------------------------------------- tables

def probe_table(models: list[str]) -> str:
    """RQ1 at the selected layer: mean ± sd over 3 seeds, 95% CI on the seed-0 test split."""
    rows = []
    for m in models:
        c, layer, cis = curves_of(m), layer_of(m), load(RESULTS / m / "probes" / "ci.json")
        n_layers = len(c["main|sandbox->sandbox|binary|last|h3"]["layer"])
        for env, env_name in (("sandbox", "Sandbox"), ("scienceworld", "ScienceWorld")):
            for setup in ("binary", "three"):
                key = f"main|{env}->{env}|{setup}|last|h3"
                ci_entry = sub(cis, f"{key}|layer{layer}")
                ctrl = at(c, f"control|{env}->{env}|{setup}|last|h3", layer)
                lex = sub(cis, f"lexical|{env}|{setup}")
                lex_min = sub(cis, f"lexical_minimal|{env}|{setup}")
                rows.append(" & ".join([
                    MODELS[m][0] if env == "sandbox" and setup == "binary" else "",
                    f"{layer}/{n_layers - 1}" if env == "sandbox" and setup == "binary" else "",
                    env_name if setup == "binary" else "", "2" if setup == "binary" else "3",
                    f"{f3(at(c, key, layer))}$\\pm${f3(at(c, key, layer, 'balanced_accuracy_std'))}",
                    f"[{f2(num(ci_entry['ci_low']))}, {f2(num(ci_entry['ci_high']))}]",
                    f3(at(c, key, layer, "minimal_balanced_accuracy_mean")),
                    f3(at(c, key, 0)), f3(num(lex["balanced_accuracy"])), f3(num(lex_min["balanced_accuracy"])),
                    f3(at(c, f"shuffled|{env}->{env}|{setup}|last|h3", layer)), f3(ctrl), f3(at(c, key, layer) - ctrl),
                    f3(at(c, key, layer, "auroc_mean")),
                ]) + r" \\")
        rows.append(r"\midrule")
    body = "\n".join(rows[:-1])
    return rf"""\begin{{table*}}[t]
\centering\small\setlength{{\tabcolsep}}{{3.2pt}}
\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{llllcccccccccc}}
\toprule
Model & Layer & Env. & Cl. & Probe bal.\ acc. & 95\% CI & Minimal pairs & Layer 0 & Lexical & Lex.\ (min.) & Shuffled & Control & Select. & AUROC \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\caption{{RQ1 probes at the layer chosen on validation data (index among decoder outputs). Balanced accuracy, mean$\pm$sd over three scenario splits; 95\% bootstrap CI on the seed-0 test split. Lexical = bag-of-words on the action text; Lex.\ (min.) = the same on minimal-pair test rows, where the action text cannot separate the classes. Control = Hewitt--Liang control task (random fixed label per verb); Select.\ = probe minus control. Majority-class balanced accuracy is 0.500 (binary) and 0.333 (three classes) by definition. Test sizes: sandbox {{\TestNSandboxBinary}} (binary) and {{\TestNSandboxThree}} (three-class) pairs, ScienceWorld {{\TestNScienceWorldBinary}} and {{\TestNScienceWorldThree}}.}}
\label{{tab:probes}}
\end{{table*}}
"""


def ablation_table(models: list[str]) -> str:
    """Brief section 7.3: every ablation at the selected layer (balanced accuracy, mean over seeds)."""
    specs = [
        ("Main probe, sandbox (binary)", "main|sandbox->sandbox|binary|last|h3", "balanced_accuracy_mean"),
        ("Token position: mean over action tokens", "token_mean|sandbox->sandbox|binary|mean|h3", "balanced_accuracy_mean"),
        ("Train on minimal pairs only", "minimal_only|sandbox->sandbox|binary|last|h3", "balanced_accuracy_mean"),
        ("Main probe, ScienceWorld (binary)", "main|scienceworld->scienceworld|binary|last|h3", "balanced_accuracy_mean"),
        ("ScienceWorld, train on minimal pairs only", "minimal_only|scienceworld->scienceworld|binary|last|h3", "balanced_accuracy_mean"),
        ("Transfer sandbox $\\rightarrow$ ScienceWorld", "transfer|sandbox->scienceworld|binary|last|h3", "balanced_accuracy_mean"),
        ("Transfer ScienceWorld $\\rightarrow$ sandbox", "transfer|scienceworld->sandbox|binary|last|h3", "balanced_accuracy_mean"),
        ("Horizon 1 labels (ScienceWorld)", "horizon|scienceworld->scienceworld|binary|last|h1", "balanced_accuracy_mean"),
        ("Horizon 2 labels (ScienceWorld)", "horizon|scienceworld->scienceworld|binary|last|h2", "balanced_accuracy_mean"),
        ("Horizon 3 labels (ScienceWorld)", "main|scienceworld->scienceworld|binary|last|h3", "balanced_accuracy_mean"),
        ("Three classes, sandbox", "main|sandbox->sandbox|three|last|h3", "balanced_accuracy_mean"),
        ("Three classes, both environments", "main|all->all|three|last|h3", "balanced_accuracy_mean"),
    ]
    rows = []
    for label, key, metric in specs:
        cells = []
        for m in models:
            c, layer = curves_of(m), layer_of(m)
            cells.append(f"{f3(at(c, key, layer, metric))}$\\pm${f3(at(c, key, layer, metric.replace('_mean', '_std')))}")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    behaviour = []
    for m in models:
        path = RESULTS / m / "behaviour.json"
        if not path.exists():
            behaviour.append(("--", "--"))
            continue
        b = load(path)
        behaviour.append((f2(num(sub(sub(b, "neutral"), "caution_rate_irreversible_version")["rate"])),
                          f2(num(sub(sub(b, "careful"), "caution_rate_irreversible_version")["rate"]))))
    rows.append(r"\midrule")
    rows.append("Caution (irrev.\\ version), neutral prompt & " + " & ".join(n for n, _ in behaviour) + r" \\")
    rows.append("Caution (irrev.\\ version), careful prompt & " + " & ".join(c for _, c in behaviour) + r" \\")
    header = " & ".join(MODELS[m][0] for m in models)
    cols = "l" + "c" * len(models)
    return rf"""\begin{{table}}[t]
\centering\small\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{{cols}}}
\toprule
Ablation & {header} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\caption{{Ablations (brief \S7.3). Probe rows: binary balanced accuracy at each model's selected layer, mean$\pm$sd over three splits; minimal-pair-only probes are tested on all test rows. The last two rows are the strict caution rate in the irreversible task versions under the neutral and the careful system prompt; the model-size ablation is the Qwen2.5-7B versus Qwen2.5-14B column pair.}}
\label{{tab:ablations}}
\end{{table}}
"""


def behaviour_table(models: list[str], kind: str = "agent") -> str:
    name = "behaviour.json" if kind == "agent" else "toolemu_behaviour.json"
    rows = []
    for m in models:
        path = RESULTS / m / name
        if not path.exists():
            continue
        b = load(path)
        for cond in ("neutral", "careful"):
            if cond not in b:
                continue
            c = sub(b, cond)
            sens = sub(c, "reversibility_sensitivity")
            rows.append(" & ".join([
                MODELS[m][0] if cond == "neutral" else "", cond,
                f2(num(sub(c, "irreversible_action_rate")["rate"])),
                f2(num(sub(c, "caution_rate_irreversible_version")["rate"])),
                f2(num(sub(c, "over_caution_rate_reversible_version")["rate"])),
                f"{f2(num(sens['value']))} ($p{pval(num(sens['p_value']))}$)",
                f2(num(sub(c, "success_rate_irreversible_version")["rate"])),
                f2(num(sub(c, "K_states_correctly")["rate"])),
                f2(num(sub(c, "A_more_careful_in_irreversible")["rate"])),
                ci(sub(c, "blindness_gap"), "value"),
                f"{num(c['parse_failures_total']):.0f}",
            ]) + r" \\")
    n_pairs = "60 task pairs" if kind == "agent" else "54 ToolEmu case pairs"
    label = "tab:behaviour" if kind == "agent" else "tab:toolemu_behaviour"
    return rf"""\begin{{table*}}[t]
\centering\small\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{llccccccccc}}
\toprule
Model & Prompt & Irrev.\ action & Caution (irrev.) & Over-caution (rev.) & Sensitivity (McNemar) & Success (irrev.) & $K$ & $A$ & Gap $K-A$ [95\% CI] & Parse fail. \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\caption{{RQ3 behaviour on {n_pairs} (each run once per version, greedy decoding). Caution = checking settings, asking the user, making a backup or choosing the safer tool before the target action. Sensitivity = caution in the irreversible minus the reversible version, with an exact McNemar test on the paired outcomes. $K$ = fraction of pairs in which the model states that the irreversible version's action is irreversible and the reversible version's is not; $A$ = fraction in which the agent is more careful in the irreversible version; Gap = $K-A$ with a bootstrap CI over pairs. Parse failures are counted, never dropped.}}
\label{{{label}}}
\end{{table*}}
"""


def rq2_table(models: list[str]) -> str:
    rows = []
    for m in models:
        s = load(RESULTS / m / "probes" / "stated_vs_probe.json")
        for env, env_name in (("sandbox", "Sandbox"), ("scienceworld", "ScienceWorld")):
            rows.append(" & ".join([
                MODELS[m][0] if env == "sandbox" else "", env_name,
                f"{f2(num(s[f'{env}_stated_accuracy_mean']))} [{f2(num(s[f'{env}_stated_accuracy_seed0_ci_low']))}, {f2(num(s[f'{env}_stated_accuracy_seed0_ci_high']))}]",
                f2(num(s[f"{env}_kappa_stated_truth_mean"])), f2(num(s[f"{env}_kappa_stated_probe_mean"])),
                f"{100 * num(s[f'{env}_unparsed_rate']):.1f}\\%", f"{num(s[f'{env}_n_test_mean']):.0f}",
            ]) + r" \\")
    return rf"""\begin{{table}}[t]
\centering\small\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{llccccc}}
\toprule
Model & Env. & Stated acc.\ [CI] & $\kappa$ truth & $\kappa$ probe & Unparsed & $n$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\caption{{RQ2: the model asked directly whether the action can be undone (three labels, option order counterbalanced), on the test pairs. Accuracy is the mean over three splits with the seed-0 95\% CI; $\kappa$ = Cohen's kappa of the stated answer with the true label and with the three-class probe at the selected layer.}}
\label{{tab:rq2}}
\end{{table}}
"""


def rq5_table(models: list[str]) -> str:
    rows = []
    for m in models:
        path = RESULTS / m / "rq5" / "report.json"
        if not path.exists():
            continue
        r = load(path)
        for h in (1, 2, 3):
            p, g = sub(r, f"h{h}_prediction_nll"), sub(r, f"h{h}_gold_nll")
            rows.append(" & ".join([
                MODELS[m][0] if h == 1 else "", str(h),
                f"{num(p['n_irreversible']):.0f}/{num(p['n_reversible']):.0f}",
                ci(p, "auroc"), f2(num(p["within_game_auroc_mean"])), ci(g, "auroc"),
                f2(num(r["exact_match_rate"])) if h == 1 else "",
            ]) + r" \\")
    return rf"""\begin{{table}}[t]
\centering\small\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lcccccc}}
\toprule
Model & $h$ & Irr./Rev. & AUROC own pred. & Within game & AUROC gold & Exact \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\caption{{RQ5 on ByteSized32-State-Prediction ({{\RQFiveN}} transitions, 31 games). AUROC of the mean token negative log-likelihood for separating irreversible from reversible transitions (0.5 = no relation), with a bootstrap 95\% CI; ``own pred.'' scores the model's greedy next-state prediction, ``gold'' the reference next state. Within game = mean AUROC inside games that have both classes. Exact = exact-match rate of the predicted state.}}
\label{{tab:rq5}}
\end{{table}}
"""


def toolemu_table(models: list[str]) -> str:
    rows = []
    for m in models:
        path = RESULTS / m / "toolemu_report.json"
        if not path.exists():
            continue
        t = load(path)
        rows.append(" & ".join([
            MODELS[m][0], f"{f2(num(t['probe_balanced_accuracy']))} [{f2(num(t['probe_ci_low']))}, {f2(num(t['probe_ci_high']))}]",
            f2(num(t["probe_auroc"])), f2(num(t["lexical_balanced_accuracy"])),
            f2(num(t["stated_accuracy"])) if "stated_accuracy" in t else "--",
            f2(num(t["kappa_stated_probe"])) if "kappa_stated_probe" in t else "--",
        ]) + r" \\")
    return rf"""\begin{{table}}[t]
\centering\small\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lccccc}}
\toprule
Model & Probe bal.\ acc.\ [CI] & AUROC & Lexical & Stated acc. & $\kappa$ stated--probe \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\caption{{External test on 108 pairs from 54 ToolEmu curated cases rebuilt in the sandbox (binary, balanced classes). The probe and the lexical baseline are trained on all sandbox pairs and never see ToolEmu data; the probe uses each model's selected layer.}}
\label{{tab:toolemu}}
\end{{table}}
"""


# --------------------------------------------------------------------------- macros

def agreement_sentence() -> str:
    path = Path("data/agreement/agreement.json")
    if not path.exists():
        return (r"\textbf{[Human agreement pending: two annotators label data/agreement/annotator\_A.csv and "
                r"annotator\_B.csv; then run python -m irrev.agreement.]}")
    a = load(path)
    return (r"Two team members labelled 100 random pairs blind to the automatic label: Cohen's $\kappa$ = "
            f"{f2(num(a['A_vs_B']))} between them and {f2(num(a['A_vs_auto']))} and {f2(num(a['B_vs_auto']))} "
            f"with the automatic labels (raw agreement {f2(num(a['raw_agreement_A_vs_B']))}).")


def macros(models: list[str]) -> str:
    lines = [macro("AgreementSentence", agreement_sentence())]
    any_model = models[0]
    cis = load(RESULTS / any_model / "probes" / "ci.json")
    for env, name in (("sandbox", "Sandbox"), ("scienceworld", "ScienceWorld")):
        for setup, sname in (("binary", "Binary"), ("three", "Three")):
            n = num(sub(cis, f"main|{env}->{env}|{setup}|last|h3|layer{layer_of(any_model)}")["n_test"])
            lines.append(macro(f"TestN{name}{sname}", f"{n:.0f}"))
    rq5 = RESULTS / any_model / "rq5" / "report.json"
    if rq5.exists():
        lines.append(macro("RQFiveN", f"{num(load(rq5)['n']):.0f}"))
    for m in models:
        tag = MODELS[m][1]
        c, layer = curves_of(m), layer_of(m)
        n_layers = len(c["main|sandbox->sandbox|binary|last|h3"]["layer"]) - 1
        lines += [macro(f"{tag}Layer", str(layer)), macro(f"{tag}Layers", str(n_layers)),
                  macro(f"{tag}ProbeSandbox", f3(at(c, "main|sandbox->sandbox|binary|last|h3", layer))),
                  macro(f"{tag}ProbeSandboxMin", f3(at(c, "main|sandbox->sandbox|binary|last|h3", layer, "minimal_balanced_accuracy_mean"))),
                  macro(f"{tag}ProbeSW", f3(at(c, "main|scienceworld->scienceworld|binary|last|h3", layer))),
                  macro(f"{tag}ProbeThree", f3(at(c, "main|all->all|three|last|h3", layer))),
                  macro(f"{tag}TransferSbSw", f3(at(c, "transfer|sandbox->scienceworld|binary|last|h3", layer))),
                  macro(f"{tag}TransferSwSb", f3(at(c, "transfer|scienceworld->sandbox|binary|last|h3", layer)))]
        s = load(RESULTS / m / "probes" / "stated_vs_probe.json")
        lines += [macro(f"{tag}StatedSandbox", f2(num(s["sandbox_stated_accuracy_mean"]))),
                  macro(f"{tag}StatedSW", f2(num(s["scienceworld_stated_accuracy_mean"]))),
                  macro(f"{tag}KappaProbe", f2(num(s["sandbox_kappa_stated_probe_mean"])))]
        stated = RESULTS / m / "stated.jsonl"
        if stated.exists():
            answers = [json.loads(x) for x in stated.read_text(encoding="utf-8").splitlines() if x]
            pairs = [a for a in answers if a.get("kind") == "pair"]
            for env, ename in (("sandbox", "Sandbox"), ("scienceworld", "SW")):
                in_env = [a for a in pairs if a.get("environment") == env]
                for answer, aname in (("irreversible", "Irrev"), ("reversible", "Rev")):
                    share = sum(a["stated"] == answer for a in in_env) / max(1, len(in_env))
                    lines.append(macro(f"{tag}Says{aname}{ename}", f"{100 * share:.0f}"))
        bpath = RESULTS / m / "behaviour.json"
        if bpath.exists():
            b = load(bpath)
            for cond, cname in (("neutral", "Neutral"), ("careful", "Careful")):
                c2 = sub(b, cond)
                gap = sub(c2, "blindness_gap")
                lines += [macro(f"{tag}Gap{cname}", f2(num(gap["value"]))),
                          macro(f"{tag}Gap{cname}CI", f"[{f2(num(gap['ci_low']))}, {f2(num(gap['ci_high']))}]"),
                          macro(f"{tag}Sens{cname}", f2(num(sub(c2, "reversibility_sensitivity")["value"]))),
                          macro(f"{tag}K{cname}", f2(num(sub(c2, "K_states_correctly")["rate"]))),
                          macro(f"{tag}A{cname}", f2(num(sub(c2, "A_more_careful_in_irreversible")["rate"]))),
                          macro(f"{tag}Caution{cname}", f2(num(sub(c2, "caution_rate_irreversible_version")["rate"]))),
                          macro(f"{tag}Parse{cname}", f"{num(c2['parse_failures_total']):.0f}")]
            mc = sub(b, "careful_vs_neutral_mcnemar")
            lines.append(macro(f"{tag}CarefulP", pval(num(mc["p_value"]))))
        spath = RESULTS / m / "steering_behaviour.json"
        sfile = RESULTS / m / "probes" / "steering_strengths.json"
        if spath.exists() and sfile.exists():
            s_max = num(load(sfile)["s_max"])
            steer = load(spath)
            lines.append(macro(f"{tag}SMax", f"{s_max:g}"))
            for kind, kname in (("direction", "Probe"), ("random_direction", "Random")):
                for sign, sname in ((-1, "Neg"), (1, "Pos")):
                    key = f"steer_{kind}_{sign * s_max:+g}"
                    if key in steer:
                        sweep_point = sub(steer, key)
                        lines += [macro(f"{tag}Steer{kname}{sname}", f2(num(sub(sweep_point, "caution_rate_irreversible_version")["rate"]))),
                                  macro(f"{tag}Steer{kname}{sname}Success", f2(num(sub(sweep_point, "success_rate_irreversible_version")["rate"]))),
                                  macro(f"{tag}Steer{kname}{sname}Parse", f"{num(sweep_point['parse_failures_total']):.0f}")]
            if "steer_direction_+0" in steer:
                lines.append(macro(f"{tag}SteerZero", f2(num(sub(sub(steer, "steer_direction_+0"), "caution_rate_irreversible_version")["rate"]))))
        rq5p = RESULTS / m / "rq5" / "report.json"
        if rq5p.exists():
            r = load(rq5p)
            lines += [macro(f"{tag}RQFiveAUROC", f2(num(sub(r, "h3_prediction_nll")["auroc"]))),
                      macro(f"{tag}RQFiveGold", f2(num(sub(r, "h3_gold_nll")["auroc"])))]
        tpath = RESULTS / m / "toolemu_report.json"
        if tpath.exists():
            t = load(tpath)
            lines += [macro(f"{tag}ToolEmuProbe", f2(num(t["probe_balanced_accuracy"]))),
                      macro(f"{tag}ToolEmuAUROC", f2(num(t["probe_auroc"]))),
                      macro(f"{tag}ToolEmuStated", f2(num(t["stated_accuracy"])))]
        tb = RESULTS / m / "toolemu_behaviour.json"
        if tb.exists():
            neutral = sub(load(tb), "neutral")
            gap = sub(neutral, "blindness_gap")
            lines += [macro(f"{tag}ToolEmuGap", f2(num(gap["value"]))),
                      macro(f"{tag}ToolEmuGapCI", f"[{f2(num(gap['ci_low']))}, {f2(num(gap['ci_high']))}]"),
                      macro(f"{tag}ToolEmuSens", f2(num(sub(neutral, "reversibility_sensitivity")["value"]))),
                      macro(f"{tag}ToolEmuIrrAct", f2(num(sub(neutral, "irreversible_action_rate")["rate"])))]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures

def probe_figure(models: list[str], out: Path) -> None:
    fig, axes = plt.subplots(2, len(models), figsize=(2.35 * len(models), 3.6), sharey=True, squeeze=False)
    for col, m in enumerate(models):
        c, layer = curves_of(m), layer_of(m)
        cis = load(RESULTS / m / "probes" / "ci.json")
        for row, env in enumerate(("sandbox", "scienceworld")):
            ax = axes[row][col]
            for name, style, color, label in (("main", "-", COLORS[m], "probe"), ("control", "-", "#c7792b", "control task"),
                                              ("shuffled", "-", "#9a9a9a", "shuffled labels")):
                key = f"{name}|{env}->{env}|binary|last|h3"
                if key not in c:
                    continue
                xs = c[key]["layer"]
                ax.plot(xs, c[key]["balanced_accuracy_mean"], ls=style, color=color if name != "main" else "#1f5aa6", lw=1.3, label=label)
            key = f"main|{env}->{env}|binary|last|h3"
            ax.plot(c[key]["layer"], c[key]["minimal_balanced_accuracy_mean"], ls="--", color="#1f5aa6", lw=1.0, label="probe, minimal pairs")
            ax.axhline(num(sub(cis, f"lexical|{env}|binary")["balanced_accuracy"]), color="#444444", ls="-.", lw=0.9, label="lexical")
            ax.axhline(num(sub(cis, f"lexical_minimal|{env}|binary")["balanced_accuracy"]), color="#444444", ls=":", lw=0.9,
                       label="lexical, minimal pairs")
            ax.axvline(layer, color="#bbbbbb", lw=0.6)
            ax.set_ylim(0.4, 1.02)
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(MODELS[m][0], fontsize=8)
            if col == 0:
                ax.set_ylabel(("Sandbox" if env == "sandbox" else "ScienceWorld") + "\nbal. acc.", fontsize=7)
            if row == 1:
                ax.set_xlabel("layer", fontsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def comparison_figure(models: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        s = load(RESULTS / m / "probes" / "stated_vs_probe.json")
        c, layer = curves_of(m), layer_of(m)
        bpath = RESULTS / m / "behaviour.json"
        sens = num(sub(sub(load(bpath), "neutral"), "reversibility_sensitivity")["value"]) if bpath.exists() else float("nan")
        values = [num(s["sandbox_stated_accuracy_mean"]), at(c, "main|sandbox->sandbox|three|last|h3", layer), sens]
        ax.bar([i + (k - (len(models) - 1) / 2) * width for i in range(3)], values, width, label=MODELS[m][0], color=COLORS[m])
    ax.axhline(1 / 3, color="#444444", ls=":", lw=0.8)
    ax.set_xticks(range(3), ["stated\naccuracy", "probe bal.\naccuracy", "behaviour\nsensitivity"], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def gap_figure(models: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    xs, labels = 0, []
    for kind, fname in (("main", "behaviour.json"), ("ToolEmu", "toolemu_behaviour.json")):
        for m in models:
            path = RESULTS / m / fname
            if not path.exists():
                continue
            b = load(path)
            for cond, marker in (("neutral", "o"), ("careful", "s")):
                if cond not in b:
                    continue
                g = sub(sub(b, cond), "blindness_gap")
                v = num(g["value"])
                ax.errorbar([xs], [v], yerr=[[v - num(g["ci_low"])], [num(g["ci_high"]) - v]], fmt=marker, color=COLORS[m], capsize=2, ms=4)
                short = {"qwen7b": "Q7B", "llama8b": "L8B", "qwen14b": "Q14B"}[m]
                labels.append(f"{short} {'N' if cond == 'neutral' else 'C'}{' TE' if kind == 'ToolEmu' else ''}")
                xs += 1
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xticks(range(xs), labels, fontsize=6, rotation=60, ha="right")
    ax.set_ylabel("gap $K-A$ (95% CI)", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def steering_figure(models: list[str], out: Path) -> None:
    present = [m for m in models if (RESULTS / m / "steering_behaviour.json").exists()]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 2.2), sharey=True, squeeze=False)
    pattern = re.compile(r"steer_(direction|random_direction)_([+-]?\d+(?:\.\d+)?)")
    for col, m in enumerate(present):
        report = load(RESULTS / m / "steering_behaviour.json")
        ax = axes[0][col]
        for kind, color in (("direction", "#1f5aa6"), ("random_direction", "#9a9a9a")):
            pts = []
            for name, metrics in report.items():
                match = pattern.fullmatch(name)
                if match and (match.group(1) == kind or (kind == "random_direction" and name == "steer_direction_+0")):
                    caution = sub(metrics, "caution_rate_irreversible_version")
                    pts.append((float(match.group(2)), num(caution["rate"]), num(caution["ci_low"]), num(caution["ci_high"]),
                                num(sub(metrics, "success_rate_irreversible_version")["rate"])))
            pts.sort()
            xs = [p[0] for p in pts]
            ax.errorbar(xs, [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], marker="o", ms=3,
                        color=color, capsize=2, lw=1.1, label=f"caution ({'probe' if kind == 'direction' else 'random'})")
            ax.plot(xs, [p[4] for p in pts], marker="s", ms=2.5, ls="--", color=color, lw=0.9,
                    label=f"success ({'probe' if kind == 'direction' else 'random'})")
        ax.set_title(MODELS[m][0], fontsize=8)
        ax.set_xlabel("strength", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("rate, irreversible versions", fontsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = 4 if len(present) >= 3 else 2
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.12 if ncol == 4 else 0.22, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = available()
    outputs = {
        "macros.tex": macros(models), "tab_probes.tex": probe_table(models), "tab_ablations.tex": ablation_table(models),
        "tab_behaviour.tex": behaviour_table(models), "tab_toolemu_behaviour.tex": behaviour_table(models, "toolemu"),
        "tab_rq2.tex": rq2_table(models), "tab_rq5.tex": rq5_table(models), "tab_toolemu.tex": toolemu_table(models),
    }
    for name, text in outputs.items():
        (GENERATED / name).write_text(text, encoding="utf-8")
    for ext in ("pdf", "png"):  # PDF for the paper, PNG for the slides
        probe_figure(models, FIGURES / f"probes.{ext}")
        comparison_figure(models, FIGURES / f"comparison.{ext}")
        gap_figure(models, FIGURES / f"gap.{ext}")
        steering_figure(models, FIGURES / f"steering.{ext}")
    print(f"models {models}: wrote {', '.join(outputs)} and 4 figures (pdf, png)")


if __name__ == "__main__":
    main()
