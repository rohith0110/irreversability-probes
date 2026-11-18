"""Recorded demo (brief section 8): replay one saved agent run step by step, or both versions of a task.

    python -m irrev.demo --model qwen7b --task free_space_00            # reversible, then irreversible
    python -m irrev.demo --model qwen7b --task free_space_00 --delay 1.5 --markdown demo.md

Every line comes from the saved run (results/<model>/agent/<condition>/episodes.jsonl): the
state the model saw, each raw model output, the tool result and the ground-truth label of
the action in that state. Nothing is generated during the demo.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

VERSIONS = ("reversible", "irreversible")


def load_episode(runs: Path, key: str) -> dict[str, list[dict[str, str]]]:
    for line in (runs / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["key"] == key:
            episode: dict[str, list[dict[str, str]]] = {"messages": record["messages"], "steps": record["steps"]}
            return episode
    raise SystemExit(f"no episode {key} in {runs / 'episodes.jsonl'}")


def transcript(runs: Path, task: str, version: str) -> list[str]:
    episode = load_episode(runs, f"{task}::{version}")
    lines = [f"## {task} ({version} version)", "", "What the agent sees:", "", "```", episode["messages"][1]["content"], "```", ""]
    for step in episode["steps"]:
        lines.append(f"**Step {int(step['index']) + 1}.** Model output: `{str(step['raw_output']).strip()}`")
        if step["parse_error"]:
            lines.append(f"- parse failure: {step['parse_error']}")
        else:
            lines.append(f"- ground-truth label of this action here: **{step['label']}**")
            lines.append(f"- tool result: {step['observation']}")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="qwen7b")
    parser.add_argument("--task", default="free_space_00")
    parser.add_argument("--condition", default="neutral")
    parser.add_argument("--versions", nargs="*", default=list(VERSIONS), choices=VERSIONS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between printed lines (live narration)")
    parser.add_argument("--markdown", type=Path, default=None, help="also write the transcript to this file")
    args = parser.parse_args()
    runs = args.results / args.model / "agent" / args.condition
    lines = [f"# Agent demo: {args.model}, {args.condition} prompt", ""]
    for version in args.versions:
        lines += transcript(runs, args.task, version)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    for line in lines:
        print(line, flush=True)
        if args.delay and line:
            time.sleep(args.delay)
    if args.markdown is not None:
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
