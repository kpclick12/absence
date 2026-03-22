from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import AppConfig, load_config
from .evaluation import binary_metrics, lesson_aggregate_metrics, subgroup_metrics
from .io import load_inputs
from .models import TrainedModel, refit_selected_model, train_task_model
from .preprocessing import prepare_data
from .reporting import write_json_report, write_markdown_report
from .scoring import build_class_scoring_frames
from .tasks import TaskDataset, build_all_task_datasets
from .utils import ensure_directory, write_frame


def _task_frames_for_experiment(task: TaskDataset, experiment_years: list[str]) -> pd.DataFrame:
    return task.frame[task.frame["academic_year"].isin(experiment_years)].reset_index(drop=True)


def _evaluate_model(
    task: TaskDataset,
    model: TrainedModel,
    frame: pd.DataFrame,
    split_name: str,
    fractions: list[float],
) -> tuple[pd.DataFrame, dict, list[dict]]:
    scored = frame[task.id_columns + [task.target_column]].copy()
    score_column = f"{task.name}_prob"
    scored[score_column] = model.predict_proba(frame)

    metrics = binary_metrics(scored, task.target_column, score_column, fractions)
    subgroup = subgroup_metrics(scored, task.primary_group_column, task.target_column, score_column, fractions)
    if task.name == "lesson":
        aggregate = lesson_aggregate_metrics(
            scored.rename(columns={score_column: "score"}),
            "score",
            task.target_column,
            fractions,
        )
        metrics.update(aggregate.metrics)
    metrics["split"] = split_name
    metrics["candidate"] = model.candidate_name
    return scored, metrics, subgroup


def run_pipeline(config_path: str | Path) -> list[dict]:
    config = load_config(config_path)
    attendance, schedule, calendar = load_inputs(config.data)
    prepared = prepare_data(attendance, schedule, calendar, config)
    tasks = build_all_task_datasets(prepared, config)

    reports_dir = ensure_directory(config.output_dir / "reports")
    predictions_dir = ensure_directory(config.output_dir / "predictions")
    models_dir = ensure_directory(config.output_dir / "models")
    production_dir = ensure_directory(models_dir / "production")

    report_records: list[dict] = []

    for experiment in config.experiments:
        experiment_prediction_dir = ensure_directory(predictions_dir / experiment.name)
        for task_name, task in tasks.items():
            train_frame = _task_frames_for_experiment(task, experiment.train_years)
            validation_frame = _task_frames_for_experiment(task, experiment.validation_years)
            test_frame = _task_frames_for_experiment(task, experiment.test_years)
            if train_frame.empty or validation_frame.empty:
                continue

            trained, candidate_records = train_task_model(task, train_frame, validation_frame, config)
            validation_scored, validation_metrics, validation_subgroups = _evaluate_model(
                task, trained, validation_frame, "validation", config.modeling.top_k_fractions
            )
            validation_metrics["experiment"] = experiment.name
            validation_metrics["task"] = task_name
            report_records.append(validation_metrics)
            report_records.extend(
                [{"experiment": experiment.name, "task": task_name, "split": "validation_subgroup", **record} for record in validation_subgroups]
            )

            write_frame(validation_scored, experiment_prediction_dir / f"{task_name}_validation.parquet")

            if not test_frame.empty:
                test_scored, test_metrics, test_subgroups = _evaluate_model(
                    task, trained, test_frame, "test", config.modeling.top_k_fractions
                )
                test_metrics["experiment"] = experiment.name
                test_metrics["task"] = task_name
                report_records.append(test_metrics)
                report_records.extend(
                    [{"experiment": experiment.name, "task": task_name, "split": "test_subgroup", **record} for record in test_subgroups]
                )
                write_frame(test_scored, experiment_prediction_dir / f"{task_name}_test.parquet")

            trained.dump(experiment_prediction_dir / f"{task_name}_{trained.candidate_name}.joblib")
            for candidate_record in candidate_records:
                report_records.append(
                    {
                        "experiment": experiment.name,
                        "task": task_name,
                        "split": "candidate_validation",
                        **candidate_record,
                    }
                )

            if experiment.production_refit_years:
                refit_frame = _task_frames_for_experiment(task, experiment.production_refit_years)
                if not refit_frame.empty:
                    production_model = refit_selected_model(task, trained.candidate_name, refit_frame, validation_frame, config)
                    production_model.dump(production_dir / f"{task_name}.joblib")

    write_json_report(report_records, reports_dir / "metrics.json")
    write_markdown_report(report_records, reports_dir / "metrics.md")
    return report_records


def score_class(
    config_path: str | Path,
    as_of_date: str,
    school_id: str,
    class_id: str,
) -> dict[str, Path]:
    config = load_config(config_path)
    attendance, schedule, calendar = load_inputs(config.data)
    prepared = prepare_data(attendance, schedule, calendar, config)
    scoring_frames = build_class_scoring_frames(prepared, config, as_of_date, school_id, class_id)

    production_dir = config.output_dir / "models" / "production"
    short_horizon_model = TrainedModel.load(production_dir / "short_horizon.joblib")
    chronic_10_model = TrainedModel.load(production_dir / "chronic_10.joblib")
    chronic_20_model = TrainedModel.load(production_dir / "chronic_20.joblib")
    chronic_year_10_model = TrainedModel.load(production_dir / "chronic_year_10.joblib")
    chronic_year_20_model = TrainedModel.load(production_dir / "chronic_year_20.joblib")
    lesson_model = TrainedModel.load(production_dir / "lesson.joblib")

    short_horizon_predictions = scoring_frames.short_horizon[
        ["date", "student_id", "school_id", "class_id", "grade", "stage"]
    ].copy()
    short_horizon_predictions["short_horizon_absence_prob"] = short_horizon_model.predict_proba(scoring_frames.short_horizon)
    short_horizon_predictions = short_horizon_predictions.sort_values("short_horizon_absence_prob", ascending=False)

    chronic_predictions = scoring_frames.chronic[["date", "student_id", "school_id", "class_id", "grade", "stage"]].copy()
    chronic_predictions["risk_10pct_term_end"] = chronic_10_model.predict_proba(scoring_frames.chronic)
    chronic_predictions["risk_20pct_term_end"] = chronic_20_model.predict_proba(scoring_frames.chronic)
    chronic_predictions = chronic_predictions.sort_values("risk_20pct_term_end", ascending=False)

    chronic_lasyar_predictions = scoring_frames.chronic_lasyar[["date", "student_id", "school_id", "class_id", "grade", "stage"]].copy()
    chronic_lasyar_predictions["risk_10pct_year_end"] = chronic_year_10_model.predict_proba(scoring_frames.chronic_lasyar)
    chronic_lasyar_predictions["risk_20pct_year_end"] = chronic_year_20_model.predict_proba(scoring_frames.chronic_lasyar)
    chronic_lasyar_predictions = chronic_lasyar_predictions.sort_values("risk_20pct_year_end", ascending=False)

    lesson_predictions = scoring_frames.lesson[
        ["score_date", "target_date", "student_id", "school_id", "class_id", "grade", "stage", "lesson_id", "target_subject", "target_lesson_start_minutes"]
    ].copy()
    lesson_predictions["lesson_absence_prob"] = lesson_model.predict_proba(scoring_frames.lesson)
    lesson_predictions = lesson_predictions.sort_values(["target_date", "target_lesson_start_minutes", "lesson_absence_prob"], ascending=[True, True, False])

    lesson_expected_absence = (
        lesson_predictions.groupby(["target_date", "lesson_id", "target_subject", "target_lesson_start_minutes"], as_index=False)[
            "lesson_absence_prob"
        ]
        .sum()
        .rename(columns={"lesson_absence_prob": "lesson_expected_absence"})
        .sort_values(["target_date", "target_lesson_start_minutes", "lesson_id"])
    )

    scoring_dir = ensure_directory(config.output_dir / "scoring" / f"{school_id}_{class_id}_{pd.Timestamp(as_of_date).date()}")
    outputs = {
        "short_horizon_predictions": scoring_dir / "short_horizon_predictions.parquet",
        "chronic_predictions": scoring_dir / "chronic_predictions.parquet",
        "chronic_lasyar_predictions": scoring_dir / "chronic_lasyar_predictions.parquet",
        "lesson_predictions": scoring_dir / "lesson_predictions.parquet",
        "lesson_expected_absence": scoring_dir / "lesson_expected_absence.parquet",
    }
    write_frame(short_horizon_predictions, outputs["short_horizon_predictions"])
    write_frame(chronic_predictions, outputs["chronic_predictions"])
    write_frame(chronic_lasyar_predictions, outputs["chronic_lasyar_predictions"])
    write_frame(lesson_predictions, outputs["lesson_predictions"])
    write_frame(lesson_expected_absence, outputs["lesson_expected_absence"])
    return outputs
