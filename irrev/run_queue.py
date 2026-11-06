"""Unattended job queue for one A30 node (run inside tmux).

Each job is a subprocess with its own log. A failed job is logged and the queue moves on.
Jobs resume from their own checkpoints, so rerunning the queue repeats nothing finished.
A heartbeat file is rewritten every minute; results/ and logs/ are uploaded to a private
Hugging Face dataset repo every hour and once more at the end. Nothing is ever deleted.

Node plans (three E2E accounts, about 24 h each):
  qwen   - Qwen2.5-7B: hidden states, layer selection, RQ2, RQ3 neutral and careful, RQ4 sweep
  llama  - Llama-3.1-8B: hidden states, layer selection, RQ2, RQ3 neutral and careful
  smoke  - Qwen2.5-0.5B on a few items: proves the queue end to end in minutes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path




def model_jobs(alias: str, root: Path, limit_args: list[str], tasks: list[str], steer: bool,
               strengths: list[str] | None = None) -> list[tuple[str, list[str]]]:
    hidden, probes = root / "hidden", root / "probes"
    task_args = ["--tasks", *tasks] if tasks else []
    jobs = [
        ("gpu_check", ["-c", "import torch, json; from irrev.models import run_metadata; "
                             f"assert torch.cuda.is_available(), 'no GPU'; print(json.dumps(run_metadata('{alias}'), indent=1))"]),
        ("hidden", ["-m", "irrev.hidden", "--model", alias, "--out", str(hidden), *limit_args]),
        ("select_layer", ["-m", "irrev.probes", "select-layer", "--hidden", str(hidden), "--out", str(probes)]),
        ("stated", ["-m", "irrev.stated", "--model", alias, "--out", str(root / "stated.jsonl"), *limit_args]),
        ("agent_neutral", ["-m", "irrev.agent", "--model", alias, "--style", "neutral", "--out", str(root / "agent" / "neutral"), "--max-tokens", "16000", *task_args]),
        ("agent_careful", ["-m", "irrev.agent", "--model", alias, "--style", "careful", "--out", str(root / "agent" / "careful"), "--max-tokens", "16000", *task_args]),
    ]
    jobs.append(("rq5_nll", ["-m", "irrev.rq5", "nll", "--model", alias, "--out", str(root / "rq5" / "nll.jsonl"),
                             *(["--limit", "20"] if limit_args else [])]))
    if steer:
        sweep = ["--strengths", *strengths] if strengths else ["--calibrate"]
        jobs.append(("steering", ["-m", "irrev.agent", "--model", alias, "--style", "neutral", "--probe-dir",
                                  str(probes), "--out", str(root / "steering"), "--max-tokens", "16000", *sweep, *task_args]))
    return jobs


def plan(name: str) -> list[tuple[str, list[str]]]:
    if name == "qwen":
        return model_jobs("qwen7b", Path("results/qwen7b"), [], [], steer=True)
    if name == "llama":
        return model_jobs("llama8b", Path("results/llama8b"), [], [], steer=False)
    if name == "smoke":
        return model_jobs("qwen0.5b", Path("results/dryrun_qwen0.5b"), ["--limit", "60"],
                          ["free_space_00", "pay_invoice_00"], steer=True, strengths=["0", "8"])
    if name == "smoke_llama":
        return model_jobs("llama1b", Path("results/dryrun_llama1b"), ["--limit", "60"],
                          ["free_space_00", "pay_invoice_00"], steer=True, strengths=["0", "8"])
    raise ValueError(f"unknown plan {name}")


class Heartbeat:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state: dict[str, str] = {}
        self.stop = threading.Event()

    def update(self, **values: str) -> None:
        self.state.update(values)
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state | {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
        log = self.state.get("log")
        if log and Path(log).exists():
            lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
            payload["last_log_line"] = lines[-1][-300:] if lines else ""
        self.path.write_text(json.dumps(payload, indent=1))

    def loop(self) -> None:
        while not self.stop.wait(60):
            self.write()


def sync(repo: str | None, folders: list[str]) -> str:
    if not repo:
        return "sync disabled"
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for folder in folders:
        if Path(folder).exists():
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, path_in_repo=folder)
    return f"synced {folders} to {repo}"


def sync_loop(repo: str | None, folders: list[str], stop: threading.Event, log: Path) -> None:
    while not stop.wait(3600):
        try:
            message = sync(repo, folders)
        except Exception as exc:  # a failed upload must not kill the run; it is retried next hour
            message = f"sync failed: {exc!r}"
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one node's job queue.")
    parser.add_argument("plan", choices=("qwen", "llama", "smoke", "smoke_llama"))
    parser.add_argument("--sync-repo", default=os.environ.get("IRREV_SYNC_REPO"))
    parser.add_argument("--only", nargs="*", default=None, help="run only these job names")
    args = parser.parse_args()
    logs = Path("logs") / args.plan
    logs.mkdir(parents=True, exist_ok=True)
    heartbeat = Heartbeat(logs / "heartbeat.json")
    threading.Thread(target=heartbeat.loop, daemon=True).start()
    folders = ["results", "logs"]
    threading.Thread(target=sync_loop, args=(args.sync_repo, folders, heartbeat.stop, logs / "sync.log"), daemon=True).start()

    status: dict[str, str] = {}
    for name, argv in plan(args.plan):
        if args.only and name not in args.only:
            continue
        log = logs / f"{name}.log"
        heartbeat.update(plan=args.plan, job=name, log=str(log), started=time.strftime("%H:%M:%S"))
        start = time.time()
        try:
            with log.open("a", encoding="utf-8") as f:
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
                f.flush()
                code = subprocess.run([sys.executable, *argv], stdout=f, stderr=subprocess.STDOUT, check=False).returncode
            status[name] = f"exit {code} after {(time.time() - start) / 60:.1f} min"
        except Exception:  # the queue must reach the next job whatever happens here
            status[name] = "crashed: " + traceback.format_exc(limit=3)
        (logs / "status.json").write_text(json.dumps(status, indent=1))
        print(f"{name}: {status[name]}", flush=True)
    heartbeat.update(job="done")
    try:
        print(sync(args.sync_repo, folders))
    except Exception as exc:  # final sync failure is reported, the data stay on disk
        print(f"final sync failed: {exc!r}")
    heartbeat.stop.set()


if __name__ == "__main__":
    main()
