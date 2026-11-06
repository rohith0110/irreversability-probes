"""Kappa computation for the human agreement sheet."""

from irrev.agreement import kappas


def test_kappas_perfect_and_partial() -> None:
    auto = {"1": "reversible", "2": "irreversible", "3": "partially_reversible", "4": "irreversible"}
    a = dict(auto)
    b = {**auto, "4": "reversible"}
    k = kappas(a, b, auto)
    assert k["A_vs_auto"] == 1.0 and k["A_vs_B"] < 1.0 and k["n"] == 4 and k["raw_agreement_A_vs_B"] == 0.75
