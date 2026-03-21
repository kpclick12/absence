from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline, score_class


def main() -> None:
    parser = argparse.ArgumentParser(prog="absence-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Train, evaluate, and save model artifacts.")
    run_parser.add_argument("--config", required=True, type=Path)

    score_parser = subparsers.add_parser("score-class", help="Score one class with previously trained production models.")
    score_parser.add_argument("--config", required=True, type=Path)
    score_parser.add_argument("--as-of-date", required=True)
    score_parser.add_argument("--school-id", required=True)
    score_parser.add_argument("--class-id", required=True)

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args.config)
        return

    outputs = score_class(args.config, args.as_of_date, args.school_id, args.class_id)
    for name, path in outputs.items():
        print(f"{name}: {path}")
