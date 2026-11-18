#!/usr/bin/env python3
"""Run the local exam-prediction pipeline in one explicit sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--lecture", type=Path)
    parser.add_argument("--slides", type=Path)
    parser.add_argument("--transcript", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    try:
        reference_dir = work / "reference_analysis"
        reference_command = [
            sys.executable, str(root / "reference_analysis.py"),
            "--clusters", str(args.clusters),
            "--reference", str(args.reference),
            "--output-dir", str(reference_dir),
        ]
        if args.slides:
            reference_command += ["--slides", str(args.slides)]
        run(reference_command)

        if args.lecture and args.slides and args.transcript:
            lecture_dir = work / "lecture_mapping"
            run([
                sys.executable, str(root / "lecture_cluster_mapping.py"),
                "--clusters", str(args.clusters),
                "--slides", str(args.slides),
                "--transcript", str(args.transcript),
                "--output-dir", str(lecture_dir),
            ])
            run([
                sys.executable, str(root / "predictor.py"),
                "--clusters", str(args.clusters),
                "--lecture", str(lecture_dir / "cluster_evidence.csv"),
                "--reference", str(reference_dir / "cluster_comparison.csv"),
                "--output-dir", str(work / "prediction"),
            ])
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: pipeline stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
