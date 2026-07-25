"""One-click project pipeline runner for standalone use."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], dry_run: bool) -> None:
    pretty = " ".join(command)
    print(f"[pipeline] {pretty}")
    if dry_run:
        return

    subprocess.run(command, check=True)


def build_steps(full: bool) -> list[list[str]]:
    python = sys.executable

    steps: list[list[str]] = []

    if full:
        steps.append([python, "scripts/train.py"])

    steps.extend(
        [
            [python, "-m", "scripts.evaluate_experiment2"],
            [python, "scripts/analysis/statistical_analysis.py"],
            [python, "-m", "scripts.analysis.manuscript_statistical_tables"],
            [python, "scripts/plot_reaching_schematic.py"],
            [python, "-m", "scripts.analysis.manuscript_behavioral_figures"],
            [python, "-m", "scripts.analysis.neural_analysis"],
            [python, "-m", "scripts.analysis.manuscript_neural_figures"],
        ]
    )

    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the NeuroRL reproducibility pipeline with one command.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include training before evaluation and analysis.",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip pytest before running pipeline stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    if Path.cwd() != repo_root:
        print(f"[pipeline] changing directory to {repo_root}")
        if not args.dry_run:
            os.chdir(repo_root)

    if not args.no_tests:
        _run([sys.executable, "-m", "pytest", "-q"], args.dry_run)

    for step in build_steps(full=args.full):
        _run(step, args.dry_run)

    print("[pipeline] completed")


if __name__ == "__main__":
    main()
