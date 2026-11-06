"""Modal runner for debugging and the pre-A30 smoke test (never for reported A30 results).

Usage (from the repo root, inside the venv):
    modal run modal_app.py --plan smoke7b          # Qwen2.5-7B smoke test on one L4
    modal volume get adnn-results results/modal ./results/modal

Weights are cached in the ``adnn-hf-cache`` volume; outputs go to ``adnn-results`` and
are committed every two minutes, because Modal GPUs are preemptible.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import modal

app = modal.App("adnn-irreversibility")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": "/cache/hf", "IRREV_HOST": "modal", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("irrev")
    .add_local_dir("data", remote_path="/root/adnn/data", ignore=["external/**", "raw/*.flawed_v1*"])
    .add_local_dir("configs", remote_path="/root/adnn/configs")
)
cache = modal.Volume.from_name("adnn-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("adnn-results", create_if_missing=True)

SMOKE_ROOT = "results/modal/smoke_qwen7b"
RESMOKE_ROOT = "results/modal/resmoke_qwen7b"
RESMOKE2_ROOT = "results/modal/resmoke2_qwen7b"
SMOKE_TASKS = ["free_space_00", "free_space_04", "clean_drafts_00", "update_config_00", "fix_document_00",
               "replace_file_00", "send_announcement_00", "reply_customer_00", "pay_invoice_00", "reimburse_00",
               "update_config_03", "pay_invoice_03"]
STEER_TASKS = ["free_space_00", "update_config_00", "send_announcement_00", "pay_invoice_00"]
PLANS: dict[str, list[list[str]]] = {
    "smoke7b": [
        ["-c", "import json; from irrev.models import run_metadata; print(json.dumps(run_metadata('qwen7b'), indent=1))"],
        ["-m", "irrev.hidden", "--model", "qwen7b", "--pairs", "data/smoke_pairs.jsonl", "--out", f"{SMOKE_ROOT}/hidden",
         "--max-tokens", "8000"],
        ["-m", "irrev.probes", "select-layer", "--pairs", "data/smoke_pairs.jsonl", "--hidden", f"{SMOKE_ROOT}/hidden",
         "--out", f"{SMOKE_ROOT}/probes"],
        ["-m", "irrev.stated", "--model", "qwen7b", "--pairs", "data/smoke_pairs.jsonl", "--limit", "200",
         "--out", f"{SMOKE_ROOT}/stated.jsonl"],
        ["-m", "irrev.agent", "--model", "qwen7b", "--out", f"{SMOKE_ROOT}/agent/neutral", "--tasks", *SMOKE_TASKS,
         "--max-tokens", "16000"],
        ["-m", "irrev.agent", "--model", "qwen7b", "--out", f"{SMOKE_ROOT}/steer_calibration", "--probe-dir",
         f"{SMOKE_ROOT}/probes", "--tasks", *STEER_TASKS, "--max-tokens", "16000",
         "--strengths", "-16", "-8", "-4", "4", "8", "16", "32"],
    ],
    "resmoke7b": [
        ["-m", "irrev.hidden", "--model", "qwen7b", "--pairs", "data/smoke_pairs.jsonl", "--out", f"{RESMOKE_ROOT}/hidden",
         "--max-tokens", "8000"],
        ["-m", "irrev.probes", "select-layer", "--pairs", "data/smoke_pairs.jsonl", "--hidden", f"{RESMOKE_ROOT}/hidden",
         "--out", f"{RESMOKE_ROOT}/probes"],
        ["-m", "irrev.stated", "--model", "qwen7b", "--pairs", "data/smoke_pairs.jsonl", "--limit", "200",
         "--out", f"{RESMOKE_ROOT}/stated.jsonl"],
        ["-m", "irrev.agent", "--model", "qwen7b", "--out", f"{RESMOKE_ROOT}/agent/neutral", "--tasks", *SMOKE_TASKS,
         "--max-tokens", "16000"],
        ["-m", "irrev.agent", "--model", "qwen7b", "--out", f"{RESMOKE_ROOT}/steer", "--probe-dir",
         f"{RESMOKE_ROOT}/probes", "--tasks", *STEER_TASKS, "--max-tokens", "16000", "--strengths", "-8", "8"],
    ],
    "resmoke2": [
        ["-m", "irrev.hidden", "--model", "qwen7b", "--pairs", "data/smoke_pairs.jsonl", "--out", f"{RESMOKE2_ROOT}/hidden",
         "--max-tokens", "8000"],
        ["-m", "irrev.probes", "select-layer", "--pairs", "data/smoke_pairs.jsonl", "--hidden", f"{RESMOKE2_ROOT}/hidden",
         "--out", f"{RESMOKE2_ROOT}/probes"],
        ["-m", "irrev.agent", "--model", "qwen7b", "--out", f"{RESMOKE2_ROOT}/agent/neutral", "--tasks", *SMOKE_TASKS,
         "--max-tokens", "16000"],
        ["-m", "irrev.agent", "--model", "qwen7b", "--out", f"{RESMOKE2_ROOT}/steer", "--probe-dir",
         f"{RESMOKE2_ROOT}/probes", "--tasks", *STEER_TASKS, "--max-tokens", "16000", "--calibrate"],
    ],
    "rq5smoke": [
        ["-m", "irrev.rq5", "nll", "--model", "qwen7b", "--limit", "64", "--out", "results/modal/rq5smoke_qwen7b/nll.jsonl"],
        ["-m", "irrev.rq5", "analyze", "--nll", "results/modal/rq5smoke_qwen7b/nll.jsonl",
         "--out", "results/modal/rq5smoke_qwen7b/report.json"],
    ],
}


def _commit_loop(stop: threading.Event) -> None:
    while not stop.wait(120):
        results.commit()


@app.function(gpu=["L4", "A10"], image=image, volumes={"/cache": cache, "/root/adnn/results": results}, timeout=4 * 3600)
def run_plan(plan: str) -> list[str]:
    os.chdir("/root/adnn")
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    report = []
    for argv in PLANS[plan]:
        start = time.time()
        code = subprocess.run([sys.executable, *argv], check=False).returncode
        report.append(f"exit {code} in {(time.time() - start) / 60:.1f} min: {' '.join(argv[:4])}")
        print(report[-1], flush=True)
        results.commit()
    stop.set()
    cache.commit()
    return report


@app.local_entrypoint()
def main(plan: str = "smoke7b") -> None:
    for line in run_plan.remote(plan):
        print(line)
