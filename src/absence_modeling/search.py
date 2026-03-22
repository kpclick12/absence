"""
Search loop runner.

Loads a pre-materialized feature parquet, applies the candidate config
(feature groups, model type, hyperparams, sample settings), trains, evaluates,
computes permutation importance, and writes compact results.

This is the inner loop of the experiment search. It is called by scripts/run_experiment.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import AppConfig
from .evaluation import binary_metrics, compute_permutation_importance, subgroup_metrics
from .feature_groups import TASK_DEFAULT_GROUPS, group_importances, resolve_features
from .models import TrainedModel, _fit_calibrators, _candidate_pipeline, _sample_training_rows
from .tasks import TaskDataset
from .utils import FeatureBundle, ensure_directory, write_frame


def load_candidate_config(candidate_dir: Path) -> dict[str, Any]:
    """Load and validate a candidate config YAML."""
    config_path = candidate_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {candidate_dir}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if "task" not in raw:
        raise ValueError("candidate config must specify 'task'")
    return raw


def _apply_row_cap(
    frame: pd.DataFrame,
    target_column: str,
    sample_fraction: float,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Sample and cap training rows for search mode."""
    if sample_fraction < 1.0:
        frame = frame.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)
    if len(frame) > max_rows:
        positive = frame[frame[target_column] == 1]
        negative = frame[frame[target_column] == 0]
        remaining = max(max_rows - len(positive), 0)
        negative = negative.sample(
            n=min(len(negative), remaining),
            random_state=seed,
            replace=False,
        )
        frame = pd.concat([positive, negative], ignore_index=True).sample(
            frac=1.0, random_state=seed
        ).reset_index(drop=True)
    return frame


def _make_task_subset(
    full_frame: pd.DataFrame,
    task_name: str,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    id_columns: list[str],
) -> TaskDataset:
    """Wrap a feature-subsetted DataFrame as a TaskDataset."""
    available_cols = set(full_frame.columns)
    missing = [f for f in numeric_features + categorical_features if f not in available_cols]
    if missing:
        raise ValueError(
            f"Feature table for '{task_name}' is missing columns: {missing}\n"
            "Re-run materialize.py after adding these features."
        )
    features = FeatureBundle(
        frame=full_frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    return TaskDataset(
        name=task_name,
        frame=full_frame,
        target_column=target_column,
        features=features,
        id_columns=id_columns,
    )


def _infer_target_column(task_name: str) -> str:
    mapping = {
        "short_horizon": "target_short_horizon",
        "chronic_10": "target_chronic_10",
        "chronic_20": "target_chronic_20",
        "chronic_year_10": "target_chronic_year_10",
        "chronic_year_20": "target_chronic_year_20",
        "lesson": "target_label",
    }
    if task_name not in mapping:
        raise ValueError(f"Unknown task: {task_name}. Valid: {sorted(mapping)}")
    return mapping[task_name]


def _infer_id_columns(task_name: str) -> list[str]:
    base = ["date", "student_id", "school_id", "class_id", "grade", "stage", "academic_year"]
    if task_name in ("chronic_10", "chronic_20", "chronic_year_10", "chronic_year_20"):
        return base + ["term_id"]
    if task_name == "lesson":
        return ["score_date", "date", "student_id", "school_id", "class_id", "grade", "stage", "lesson_id", "academic_year"]
    return base


def run_candidate(
    candidate_dir: Path,
    features_dir: Path,
    config: AppConfig,
    search_log_path: Path,
    compute_importance: bool = True,
    n_importance_repeats: int = 5,
) -> dict[str, Any]:
    """
    Run one search-mode experiment iteration.

    Loads the pre-materialized feature table for the task, applies feature group
    selection and row sampling from the candidate config, trains the specified model,
    evaluates on the validation split, computes permutation importance, and writes
    compact artifacts to candidate_dir.

    Returns the metrics dict.
    """
    t0 = time.monotonic()
    candidate_dir = Path(candidate_dir)
    candidate_config = load_candidate_config(candidate_dir)

    task_name: str = candidate_config["task"]
    feature_groups: list[str] = candidate_config.get("feature_groups", TASK_DEFAULT_GROUPS[task_name])
    model_cfg: dict[str, Any] = candidate_config.get("model", {})
    model_type: str = model_cfg.get("type", "logistic")
    hyperparams: dict[str, Any] = {k: v for k, v in model_cfg.items() if k != "type"}
    sample_fraction: float = float(candidate_config.get("sample_fraction", 1.0))
    max_rows: int = int(candidate_config.get("max_rows", config.modeling.max_train_rows_per_task))
    seed: int = int(candidate_config.get("seed", config.project.random_seed))

    # Resolve feature groups to concrete column lists
    numeric_features, categorical_features = resolve_features(feature_groups, task_name)
    target_column = _infer_target_column(task_name)
    id_columns = _infer_id_columns(task_name)

    # Load materialized feature table
    feature_table_path = features_dir / f"{task_name}.parquet"
    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_table_path}\n"
            "Run scripts/materialize.py first."
        )
    full_frame = pd.read_parquet(feature_table_path)

    # Identify train and validation years from the first experiment in config
    # (or from candidate config override)
    experiment_name = candidate_config.get("experiment", config.experiments[0].name)
    experiment = next((e for e in config.experiments if e.name == experiment_name), config.experiments[0])
    train_frame = full_frame[full_frame["academic_year"].isin(experiment.train_years)].copy()
    val_frame = full_frame[full_frame["academic_year"].isin(experiment.validation_years)].copy()

    if train_frame.empty:
        raise RuntimeError(f"Training frame is empty for years {experiment.train_years}")
    if val_frame.empty:
        raise RuntimeError(f"Validation frame is empty for years {experiment.validation_years}")

    # Sample and cap training rows
    train_sample = _apply_row_cap(train_frame, target_column, sample_fraction, max_rows, seed)

    # Build task subset with selected features only
    task = _make_task_subset(full_frame, task_name, target_column, numeric_features, categorical_features, id_columns)

    # Train
    x_columns = numeric_features + categorical_features
    pipeline = _candidate_pipeline(model_type, task, seed, hyperparams)
    pipeline.fit(train_sample[x_columns], train_sample[target_column])

    # Evaluate on validation
    val_probs = pipeline.predict_proba(val_frame[x_columns])[:, 1]
    metrics = binary_metrics(
        val_frame.assign(score=val_probs),
        target_column,
        "score",
        config.modeling.top_k_fractions,
    )
    subgroups = subgroup_metrics(
        val_frame.assign(score=val_probs),
        "stage",
        target_column,
        "score",
        config.modeling.top_k_fractions,
    )

    # Calibrate and save model artifact
    stage_calibrators, global_calibrator = _fit_calibrators(val_probs, val_frame, target_column)
    trained_model = TrainedModel(
        task_name=task_name,
        candidate_name=model_type,
        model=pipeline,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        stage_calibrators=stage_calibrators,
        global_calibrator=global_calibrator,
        metadata={"hyperparams": hyperparams, "feature_groups": feature_groups},
    )
    ensure_directory(candidate_dir)
    trained_model.dump(str(candidate_dir / "model.joblib"))

    # Permutation importance (on a capped validation subset for speed)
    importance_by_feature: dict[str, float] = {}
    importance_by_group: dict[str, float] = {}
    if compute_importance:
        importance_val = val_frame.head(min(len(val_frame), 5000))
        importance_by_feature = compute_permutation_importance(
            pipeline,
            importance_val[x_columns],
            target_column,
            x_columns,
            n_repeats=n_importance_repeats,
            random_state=seed,
        )
        importance_by_group = group_importances(importance_by_feature)

    runtime_s = round(time.monotonic() - t0, 1)

    # Write compact artifacts
    full_metrics = {
        "candidate": candidate_dir.name,
        "task": task_name,
        "experiment": experiment_name,
        "model_type": model_type,
        "hyperparams": hyperparams,
        "feature_groups": feature_groups,
        "n_features": len(numeric_features) + len(categorical_features),
        "train_rows": len(train_sample),
        "val_rows": len(val_frame),
        "runtime_s": runtime_s,
        **metrics,
        "subgroups": subgroups,
    }
    (candidate_dir / "metrics.json").write_text(json.dumps(full_metrics, indent=2))

    importance_output = {"by_feature": importance_by_feature, "by_group": importance_by_group}
    (candidate_dir / "importance.json").write_text(json.dumps(importance_output, indent=2))

    # Append one summary row to the shared search log
    _append_search_log(search_log_path, full_metrics, importance_by_group)

    return full_metrics


def _append_search_log(log_path: Path, metrics: dict[str, Any], importance_by_group: dict[str, float]) -> None:
    """Append one row to the TSV search log. Creates file with header if needed."""
    import csv

    top_groups_by_importance = sorted(importance_by_group, key=importance_by_group.get, reverse=True)[:3]  # type: ignore[arg-type]
    row = {
        "candidate": metrics["candidate"],
        "task": metrics["task"],
        "model_type": metrics["model_type"],
        "feature_groups": ",".join(metrics["feature_groups"]),
        "n_features": metrics["n_features"],
        "train_rows": metrics["train_rows"],
        "precision_at_0.01": metrics.get("precision_at_0.01", ""),
        "precision_at_0.05": metrics.get("precision_at_0.05", ""),
        "pr_auc": metrics.get("pr_auc", ""),
        "brier": metrics.get("brier", ""),
        "runtime_s": metrics["runtime_s"],
        "top_groups": ",".join(top_groups_by_importance),
    }
    write_header = not log_path.exists()
    ensure_directory(log_path.parent)
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
