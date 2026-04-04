#!/usr/bin/env python3
"""
Pipeline orchestrator for synthetic NPC training data generation.

Runs the full generate → validate → dataset pipeline in the correct order,
pausing at model swap points to wait for confirmation.

Usage:
    python3 run.py [--config config.yaml] [--resume]

State is saved to data/pipeline_state.json after each stage so --resume
can pick up where it left off if interrupted.
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Track all child processes so we can clean them up on exit
_children: list[subprocess.Popen] = []


def _cleanup_children(signum=None, frame=None) -> None:
    """Kill all child processes on SIGTERM/SIGINT."""
    for p in _children:
        if p.poll() is None:
            p.terminate()
    # Give them a moment then force kill
    time.sleep(2)
    for p in _children:
        if p.poll() is None:
            p.kill()
    sys.exit(1)


signal.signal(signal.SIGTERM, _cleanup_children)
signal.signal(signal.SIGINT, _cleanup_children)

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
# Each stage has:
#   id         — unique identifier, used for state tracking
#   desc       — what's happening
#   cmd        — command(s) to run. List of lists = parallel. List of str = sequential.
#   swap_after — if set, pause after this stage and prompt for a model swap

STAGES = [
    {
        "id": "generate_writer",
        "desc": "Stage 1 — Generate writer samples (Writer on 6900 XT)",
        "hardware": "6900 XT: Writer (Qwen3.5-27B-Writer)",
        "cmd": [["python3", "generate.py", "--model", "writer"]],
        "parallel": False,
        "swap_after": {
            "stop":  "Writer (Qwen3.5-27B-Writer)",
            "start": "Opus Distill (Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled)",
            "verify_hint": "curl -s http://192.168.1.15:8080/v1/models | python3 -m json.tool",
        },
    },
    {
        "id": "generate_opus_and_filter_writer",
        "desc": "Stage 2 — Generate opus samples + fast filter writer samples",
        "hardware": "6900 XT: Opus Distill  |  GTX 1070: OmniCoder (fast filter)",
        "cmd": [
            ["python3", "generate.py", "--model", "opus"],
            ["python3", "validate.py", "--pass", "fast-filter", "--source", "writer"],
        ],
        "parallel": True,
        "swap_after": None,  # Opus Distill stays loaded for Stage 3
    },
    {
        "id": "crossreview_writer_and_filter_opus",
        "desc": "Stage 3 — Drummer cross-reviews filtered writer + fast filter raw opus",
        "hardware": "6900 XT: Drummer fine-tune  |  GTX 1070: OmniCoder (fast filter)",
        "cmd": [
            ["python3", "validate.py", "--pass", "cross-review", "--source", "writer", "--reviewer", "drummer"],
            ["python3", "validate.py", "--pass", "fast-filter", "--source", "opus"],
        ],
        "parallel": True,
        "swap_after": {
            "stop":  "Drummer fine-tune",
            "start": "Writer (Qwen3.5-27B-Writer)",
            "verify_hint": "curl -s http://192.168.1.15:8080/v1/models | python3 -m json.tool",
        },
    },
    {
        "id": "crossreview_writer_strict",
        "desc": "Stage 3b — Drummer second pass on writer samples (stricter threshold)",
        "hardware": "6900 XT: Drummer fine-tune",
        "cmd": [["python3", "validate.py", "--pass", "cross-review", "--source", "writer",
                 "--reviewer", "drummer", "--input", "data/validated_writer.jsonl"]],
        "parallel": False,
        "swap_after": None,
    },
    {
        "id": "crossreview_opus",
        "desc": "Stage 4 — Writer cross-reviews filtered opus samples",
        "hardware": "6900 XT: Writer (Qwen3.5-27B-Writer)",
        "cmd": [
            ["python3", "validate.py", "--pass", "cross-review", "--source", "opus", "--reviewer", "writer"],
        ],
        "parallel": False,
        "swap_after": None,
    },
    {
        "id": "build_dataset",
        "desc": "Stage 5 — Merge validated samples and build final dataset",
        "hardware": "Local (no GPU needed)",
        "cmd": [["python3", "dataset.py"]],
        "parallel": False,
        "swap_after": None,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}{msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def success(msg: str) -> None:
    print(f"{GREEN}✓ {msg}{RESET}")

def warn(msg: str) -> None:
    print(f"{YELLOW}! {msg}{RESET}")

def error(msg: str) -> None:
    print(f"{RED}✗ {msg}{RESET}")


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed": []}


def save_state(state_path: Path, state: dict) -> None:
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def run_sequential(cmd: list[str], cwd: Path) -> bool:
    """Run a single command, streaming output. Returns True on success."""
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd)
    _children.append(proc)
    proc.wait()
    _children.remove(proc)
    return proc.returncode == 0


def run_parallel(cmds: list[list[str]], cwd: Path) -> bool:
    """Run multiple commands in parallel, wait for all. Returns True if all succeed."""
    print(f"  Running {len(cmds)} processes in parallel:")
    for cmd in cmds:
        print(f"    $ {' '.join(cmd)}")

    procs = [subprocess.Popen(cmd, cwd=cwd) for cmd in cmds]
    _children.extend(procs)
    print()

    # Poll until all done, showing which are still running
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


def prompt_swap(swap: dict) -> None:
    """Block until user confirms a model swap on the main PC."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed stages")
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

    print(f"\n{BOLD}D&D NPC Training Data Pipeline{RESET}")
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
