#!/usr/bin/env python3
"""Master script to run training/evaluation/backbone experiments.

The project page in docs/ is maintained independently and is never written to by this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def run_python_script(root: Path, script_rel: str, args: Iterable[str]) -> None:
    cmd = [sys.executable, str(root / script_rel), *args]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=root, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full VisionDrive3D AI pipeline")
    parser.add_argument("--dataset", type=str, default="./output_dataset")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = resolve_dataset_root(args.dataset, root)

    dataset_arg = str(dataset_root)

    if not args.skip_train:
        run_python_script(root, "src/ai_demo/train.py", ["--dataset", dataset_arg, "--batch", "32"])
    else:
        print("[INFO] Skipping training step (--skip-train)")

    run_python_script(root, "src/ai_demo/evaluate.py", ["--dataset", dataset_arg])

    if not args.skip_backbone:
        run_python_script(root, "src/ai_demo/experiment_backbones.py", ["--dataset", dataset_arg])
    else:
        print("[INFO] Skipping backbone experiment step (--skip-backbone)")

    print(f"Done. Results written to {dataset_root}.")


if __name__ == "__main__":
    main()
