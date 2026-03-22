#!/usr/bin/env python
"""
Experiment search runner.

Trains one candidate config against pre-materialized feature tables and writes
compact results. This is the inner loop of the semi-automated search process.

Usage:
    # Search mode (fast, sampled data):
    python scripts/run_experiment.py --config configs/real.yaml --candidate experiments/001/

    # Finalize mode (full data, fixed holdout, save production artifact):
    python scripts/run_experiment.py --config configs/real.yaml --candidate experiments/009/ --finalize

The candidate directory must contain a config.yaml. See configs/candidate_template.yaml for format.

After running, results are in:
    experiments/{name}/metrics.json     - full metrics
    experiments/{name}/importance.json  - permutation importance by feature and by group
    experiments/{name}/model.joblib     - fitted model artifact
    experiments/search_log.tsv          - one-row summary appended for every run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from absence_modeling.config import load_config
from absence_modeling.io import load_inputs
from absence_modeling.models import refit_selected_model
from absence_modeling.preprocessing import prepare_data
from absence_modeling.search import load_candidate_config, run_candidate
from absence_modeling.tasks import build_all_task_datasets
from absence_modeling.utils import ensure_directory


def _finalize(candidate_dir: Path, config_path: str) -> None:
    """
    Finalize mode: retrain on full production data and save artifact.
    Uses config.experiments[-1].production_refit_years if set,
    otherwise uses all data in the feature table.
    """
    config = load_config(config_path)
    candidate_config = load_candidate_config(candidate_dir)
    task_name = candidate_config["task"]

    # We need prepared data for finalize (full pipeline, not just parquet)
    print("Loading raw data for full retraining...")
    attendance, schedule, calendar = load_inputs(config.data)
    prepared = prepare_data(attendance, schedule, calendar, config)
    tasks = build_all_task_datasets(prepared, config)
    task = tasks[task_name]

    # Find the experiment with production_refit_years
    final_experiment = next(
        (e for e in reversed(config.experiments) if e.production_refit_years),
        config.experiments[-1],
    )
    refit_years = final_experiment.production_refit_years or final_experiment.train_years
    val_years = final_experiment.validation_years

    refit_frame = task.frame[task.frame["academic_year"].isin(refit_years)].copy()
    val_frame = task.frame[task.frame["academic_year"].isin(val_years)].copy()

    model_type = candidate_config.get("model", {}).get("type", "logistic")
    print(f"Retraining {task_name} / {model_type} on {len(refit_frame):,} rows...")
    production_model = refit_selected_model(task, model_type, refit_frame, val_frame, config)

    production_dir = ensure_directory(config.output_dir / "models" / "production")
    production_model.dump(str(production_dir / f"{task_name}.joblib"))
    print(f"Production artifact saved: {production_dir / task_name}.joblib")

    # Write a finalize summary
    summary = {
        "candidate": candidate_dir.name,
        "task": task_name,
        "model_type": model_type,
        "refit_years": refit_years,
        "refit_rows": len(refit_frame),
        "production_artifact": str(production_dir / f"{task_name}.joblib"),
    }
    (candidate_dir / "finalize_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment candidate.")
    parser.add_argument("--config", required=True, help="Path to global config YAML")
    parser.add_argument("--candidate", required=True, help="Path to candidate directory")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Finalize mode: retrain on full data and save production artifact",
    )
    parser.add_argument(
        "--no-importance",
        action="store_true",
        help="Skip permutation importance (faster runs during early search)",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Override features directory (default: <config.output_dir>/features)",
    )
    args = parser.parse_args()

    candidate_dir = Path(args.candidate)
    if not candidate_dir.exists():
        print(f"Error: candidate directory not found: {candidate_dir}")
        sys.exit(1)

    if args.finalize:
        _finalize(candidate_dir, args.config)
        return

    config = load_config(args.config)
    features_dir = Path(args.features_dir) if args.features_dir else config.output_dir / "features"
    search_log_path = candidate_dir.parent / "search_log.tsv"

    print(f"Running candidate: {candidate_dir.name}")
    metrics = run_candidate(
        candidate_dir=candidate_dir,
        features_dir=features_dir,
        config=config,
        search_log_path=search_log_path,
        compute_importance=not args.no_importance,
    )

    # Print compact summary
    k1 = f"precision_at_{config.modeling.top_k_fractions[0]:.2f}"
    k2 = f"precision_at_{config.modeling.top_k_fractions[-1]:.2f}"
    print(
        f"\n{'='*50}\n"
        f"  candidate : {metrics['candidate']}\n"
        f"  task      : {metrics['task']}\n"
        f"  model     : {metrics['model_type']}\n"
        f"  features  : {metrics['n_features']} ({','.join(metrics['feature_groups'])})\n"
        f"  train rows: {metrics['train_rows']:,}\n"
        f"  {k1}  : {metrics.get(k1, 'n/a')}\n"
        f"  {k2}  : {metrics.get(k2, 'n/a')}\n"
        f"  pr_auc    : {metrics.get('pr_auc', 'n/a')}\n"
        f"  brier     : {metrics.get('brier', 'n/a')}\n"
        f"  runtime   : {metrics['runtime_s']}s\n"
        f"{'='*50}"
    )
    print(f"\nResults: {candidate_dir}/metrics.json")
    print(f"Log:     {search_log_path}")


if __name__ == "__main__":
    main()
