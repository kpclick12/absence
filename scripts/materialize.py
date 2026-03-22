#!/usr/bin/env python
"""
Feature materialization script.

Run this ONCE after raw data arrives. Saves model-ready feature tables
for all tasks to parquet. Subsequent experiment iterations load these
parquets directly, skipping expensive raw data reprocessing.

Usage:
    python scripts/materialize.py --config configs/real.yaml

Output:
    artifacts/features/{task_name}.parquet
    artifacts/features/{task_name}_schema.json
    artifacts/features/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from absence_modeling.config import load_config
from absence_modeling.io import load_inputs
from absence_modeling.preprocessing import prepare_data
from absence_modeling.tasks import build_all_task_datasets
from absence_modeling.utils import ensure_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize feature tables for all tasks.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: <config.output_dir>/features)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    features_dir = Path(args.output_dir) if args.output_dir else config.output_dir / "features"
    ensure_directory(features_dir)

    print(f"Loading raw data from config: {args.config}")
    attendance, schedule, calendar = load_inputs(config.data)

    print("Preprocessing (joins, rolling features, prior-year lookup)...")
    prepared = prepare_data(attendance, schedule, calendar, config)

    print("Building all task datasets...")
    tasks = build_all_task_datasets(prepared, config)

    summary: dict[str, dict] = {}
    for task_name, task in tasks.items():
        frame = task.frame
        out_path = features_dir / f"{task_name}.parquet"
        frame.to_parquet(out_path, index=False)

        schema = {
            "target_column": task.target_column,
            "id_columns": task.id_columns,
            "numeric_features": task.features.numeric_features,
            "categorical_features": task.features.categorical_features,
            "columns": list(frame.columns),
            "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        }
        (features_dir / f"{task_name}_schema.json").write_text(json.dumps(schema, indent=2))

        positive_rate = (
            float(frame[task.target_column].mean())
            if task.target_column in frame.columns and len(frame) > 0
            else None
        )
        task_summary = {
            "rows": len(frame),
            "positive_rate": positive_rate,
            "n_numeric_features": len(task.features.numeric_features),
            "n_categorical_features": len(task.features.categorical_features),
            "years": sorted(frame["academic_year"].unique().tolist()) if "academic_year" in frame.columns else [],
            "parquet": str(out_path),
        }
        summary[task_name] = task_summary
        print(
            f"  {task_name:25s} {len(frame):>8,} rows  "
            f"positive={positive_rate:.3f}  "
            f"features={len(task.features.numeric_features)}+{len(task.features.categorical_features)}"
        )

    (features_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFeature tables written to: {features_dir}")
    print("Run scripts/run_experiment.py to start the search loop.")


if __name__ == "__main__":
    main()
