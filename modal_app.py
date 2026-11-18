"""Modal runner for debugging and the pre-A30 smoke test (never for reported A30 results).

Usage (from the repo root, inside the venv):
    modal run modal_app.py --plan smoke7b          # Qwen2.5-7B smoke test on one L4
    modal volume get adnn-results results/modal ./results/modal

Weights are cached in the ``adnn-hf-cache`` volume; outputs go to ``adnn-results`` and
are committed every two minutes, because Modal GPUs are preemptible.
"""

from __future__ import annotations

import os
import shutil
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
    # the node's own queue on the final data, limited to the jobs that depend on data/pairs.jsonl
    "queuecheck": [
        ["-m", "irrev.run_queue", "qwen", "--only", "gpu_check", "hidden", "select_layer", "stated"],
        ["-c", "import shutil; shutil.copytree('logs/qwen', 'results/modal/queuecheck_logs', dirs_exist_ok=True)"],
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


FINAL_ROOT = "/vol/final"


@app.function(gpu="L40S", image=image, volumes={"/cache": cache, "/vol": results},
              secrets=[modal.Secret.from_name("adnn-hf")], timeout=12 * 3600, retries=3)
def run_node_queue(queue: str) -> None:
    """The node queue itself (irrev.run_queue) writing to final/ on the volume; a retry resumes from checkpoints.

    ``queue`` is a plan name, optionally with ``:job+job`` to run only those jobs (e.g. ``llama:steering``).
    """
    plan, _, only = queue.partition(":")
    os.chdir("/root/adnn")
    os.makedirs(f"{FINAL_ROOT}/results", exist_ok=True)
    os.makedirs(f"{FINAL_ROOT}/logs", exist_ok=True)
    for name in ("results", "logs"):
        if not os.path.exists(name):
            os.symlink(f"{FINAL_ROOT}/{name}", name)
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    only_args = ["--only", *only.split("+")] if only else []
    subprocess.run([sys.executable, "-m", "irrev.run_queue", plan, *only_args], check=False)
    stop.set()
    results.commit()
    cache.commit()


@app.function(cpu=32, memory=65536, image=image, volumes={"/vol": results}, timeout=4 * 3600)
def run_analysis(commands: list[list[str]]) -> None:
    """CPU-only analysis (python -m irrev.<...>) on the results already in final/ on the volume."""
    os.chdir("/root/adnn")
    if not os.path.exists("results"):
        os.symlink(f"{FINAL_ROOT}/results", "results")
    for argv in commands:
        start = time.time()
        if "--hidden" in argv:  # memory-mapped reads over the network volume are very slow; use local disk
            i = argv.index("--hidden") + 1
            local = f"/tmp/{argv[i]}"
            if not os.path.exists(local):
                shutil.copytree(argv[i], local)
            argv = [*argv[:i], local, *argv[i + 1:]]
        # one BLAS thread per joblib worker: 32 workers x 32 threads each thrash the CPU
        single = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        code = subprocess.run([sys.executable, "-m", *argv], check=False, env=os.environ | single).returncode
        print(f"exit {code} in {(time.time() - start) / 60:.1f} min: {' '.join(argv[:3])}", flush=True)
        results.commit()


@app.local_entrypoint()
def main(plan: str = "smoke7b", queues: str = "", analysis: str = "") -> None:
    if analysis:
        run_analysis.remote([command.split() for command in analysis.split(";")])
        return
    if queues:
        list(run_node_queue.map(queues.split(",")))
        return
    for line in run_plan.remote(plan):
        print(line)
