#!/usr/bin/env python3
"""
Round 2 pipeline — Writer (300) + Drummer (300) + Opus (150) generation.

Review plan:
  - Qwen3.5-UD reviews Writer's samples
  - Writer reviews Drummer's samples
  - Writer reviews Opus's samples

Parallelism:
  - Drummer generates while OmniCoder fast-filters Writer's raw output
  - Opus generates while OmniCoder fast-filters Drummer's raw output
  - OmniCoder fast-filters Opus + Qwen cross-reviews Writer simultaneously

Usage:
    python3 run_r2.py [--config config_r2.yaml] [--resume]
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

_children: list[subprocess.Popen] = []


def _cleanup_children(signum=None, frame=None) -> None:
    for p in _children:
        if p.poll() is None:
            p.terminate()
    time.sleep(2)
    for p in _children:
        if p.poll() is None:
            p.kill()
    sys.exit(1)


signal.signal(signal.SIGTERM, _cleanup_children)
signal.signal(signal.SIGINT, _cleanup_children)

STAGES = [
    {
        "id": "generate_writer_r2",
        "desc": "Stage 1 — Generate writer samples (Writer on 6900 XT)",
        "hardware": "6900 XT: Writer (Qwen3.5-27B-Writer)",
        "cmd": [["python3", "generate.py", "--config", "config_r2.yaml", "--model", "writer"]],
        "parallel": False,
        "swap_after": {
            "stop":  "Writer (Qwen3.5-27B-Writer)",
            "start": "Drummer fine-tune (Mistral-Small-Drummer-22B)",
            "verify_hint": "curl -s http://192.168.1.15:8080/v1/models | python3 -m json.tool",
        },
    },
    {
        "id": "generate_drummer_and_filter_writer_r2",
        "desc": "Stage 2 — Drummer generates + OmniCoder fast-filters writer (parallel)",
        "hardware": "6900 XT: Drummer  |  GTX 1070: OmniCoder (fast filter)",
        "cmd": [
            ["python3", "generate.py", "--config", "config_r2.yaml", "--model", "drummer"],
            ["python3", "validate.py", "--config", "config_r2.yaml", "--pass", "fast-filter", "--source", "writer"],
        ],
        "parallel": True,
        "swap_after": {
            "stop":  "Drummer fine-tune (Mistral-Small-Drummer-22B)",
            "start": "Opus Distill (Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled)",
            "verify_hint": "curl -s http://192.168.1.15:8080/v1/models | python3 -m json.tool",
        },
    },
    {
        "id": "generate_opus_and_filter_drummer_r2",
        "desc": "Stage 3 — Opus generates + OmniCoder fast-filters drummer (parallel)",
        "hardware": "6900 XT: Opus Distill  |  GTX 1070: OmniCoder (fast filter)",
        "cmd": [
            ["python3", "generate.py", "--config", "config_r2.yaml", "--model", "opus"],
            ["python3", "validate.py", "--config", "config_r2.yaml", "--pass", "fast-filter", "--source", "drummer"],
        ],
        "parallel": True,
        "swap_after": {
            "stop":  "Opus Distill (Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled)",
            "start": "Qwen3.5-UD (Qwen3.5-27B-UD-Q3_K_XL)",
            "verify_hint": "curl -s http://192.168.1.15:8080/v1/models | python3 -m json.tool",
        },
    },
    {
        "id": "filter_opus_and_review_writer_r2",
        "desc": "Stage 4 — OmniCoder fast-filters opus + Qwen reviews writer (parallel)",
        "hardware": "6900 XT: Qwen3.5-UD  |  GTX 1070: OmniCoder (fast filter)",
        "cmd": [
            ["python3", "validate.py", "--config", "config_r2.yaml", "--pass", "fast-filter", "--source", "opus"],
            ["python3", "validate.py", "--config", "config_r2.yaml", "--pass", "cross-review", "--source", "writer", "--reviewer", "qwen"],
        ],
        "parallel": True,
        "swap_after": {
            "stop":  "Qwen3.5-UD (Qwen3.5-27B-UD-Q3_K_XL)",
            "start": "Writer (Qwen3.5-27B-Writer)",
            "verify_hint": "curl -s http://192.168.1.15:8080/v1/models | python3 -m json.tool",
        },
    },
    {
        "id": "review_drummer_r2",
        "desc": "Stage 5 — Writer cross-reviews drummer samples",
        "hardware": "6900 XT: Writer (Qwen3.5-27B-Writer)",
        "cmd": [["python3", "validate.py", "--config", "config_r2.yaml", "--pass", "cross-review", "--source", "drummer", "--reviewer", "writer"]],
        "parallel": False,
        "swap_after": None,
    },
    {
        "id": "review_opus_r2",
        "desc": "Stage 6 — Writer cross-reviews opus samples",
        "hardware": "6900 XT: Writer (Qwen3.5-27B-Writer)",
        "cmd": [["python3", "validate.py", "--config", "config_r2.yaml", "--pass", "cross-review", "--source", "opus", "--reviewer", "writer"]],
        "parallel": False,
        "swap_after": None,
    },
    {
        "id": "build_dataset_r2",
        "desc": "Stage 7 — Merge validated samples and build final dataset",
        "hardware": "Local (no GPU needed)",
        "cmd": [["python3", "dataset.py", "--config", "config_r2.yaml"]],
        "parallel": False,
        "swap_after": None,
    },
]

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(msg):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}{msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def success(msg): print(f"{GREEN}✓ {msg}{RESET}")
def warn(msg):    print(f"{YELLOW}! {msg}{RESET}")
def error(msg):   print(f"{RED}✗ {msg}{RESET}")


def load_state(state_path):
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed": []}


def save_state(state_path, state):
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def run_sequential(cmd, cwd):
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd)
    _children.append(proc)
    proc.wait()
    _children.remove(proc)
    return proc.returncode == 0


def run_parallel(cmds, cwd):
    print(f"  Running {len(cmds)} processes in parallel:")
    for cmd in cmds:
        print(f"    $ {' '.join(cmd)}")
    procs = [subprocess.Popen(cmd, cwd=cwd) for cmd in cmds]
    _children.extend(procs)
    print()
    while True:
        still_running = [i for i, p in enumerate(procs) if p.poll() is None]
        if not still_running:
            break
        labels = [" ".join(cmds[i][-3:]) for i in still_running]
        print(f"  Still running: {', '.join(labels)}", end="\r")
        time.sleep(5)
    print()
    for p in procs:
        _children.remove(p)
    results = [p.returncode for p in procs]
    failed = [i for i, r in enumerate(results) if r != 0]
    if failed:
        for i in failed:
            error(f"Process failed (exit {results[i]}): {' '.join(cmds[i])}")
        return False
    return True


def prompt_swap(swap):
    header("MODEL SWAP REQUIRED")
    print(f"\n  Stop  : {BOLD}{swap['stop']}{RESET}")
    print(f"  Start : {BOLD}{swap['start']}{RESET}")
    print(f"\n  To verify the new model is responding:")
    print(f"    {swap['verify_hint']}")
    print()
    while True:
        try:
            ans = input(f"{YELLOW}  Press Enter once the swap is complete (or 'q' to quit): {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)
        if ans == "q":
            print("Exiting. Run with --resume to continue from this point.")
            sys.exit(0)
        if ans == "":
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_r2.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    cfg_path = script_dir / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["output_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = data_dir / "pipeline_state.json"

    state = load_state(state_path)
    completed = set(state["completed"])

    print(f"\n{BOLD}D&D NPC Training Data Pipeline — Round 2{RESET}")
    print(f"Config: {cfg_path}")
    print(f"Output: {data_dir}")
    if args.resume and completed:
        warn(f"Resuming — skipping {len(completed)} completed stage(s): {', '.join(sorted(completed))}")

    for stage in STAGES:
        sid = stage["id"]
        if args.resume and sid in completed:
            success(f"[skip] {stage['desc']}")
            continue

        header(stage["desc"])
        print(f"  Hardware: {stage['hardware']}\n")

        if stage["parallel"]:
            ok = run_parallel(stage["cmd"], cwd=script_dir)
        else:
            ok = run_sequential(stage["cmd"][0], cwd=script_dir)

        if not ok:
            error(f"Stage '{sid}' failed. Fix the issue and re-run with --resume.")
            sys.exit(1)

        completed.add(sid)
        state["completed"] = list(completed)
        save_state(state_path, state)
        success(f"Stage '{sid}' complete.")

        if stage.get("swap_after"):
            prompt_swap(stage["swap_after"])

    header("Pipeline complete!")
    print(f"\n  Final dataset: {cfg.get('final_dataset', data_dir / 'dataset.jsonl')}")
    print(f"  Run axolotl to start fine-tuning.\n")


if __name__ == "__main__":
    main()
