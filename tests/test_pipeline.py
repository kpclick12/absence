from __future__ import annotations

import pandas as pd

from absence_modeling.config import load_config
from absence_modeling.io import load_inputs
from absence_modeling.pipeline import run_pipeline, score_class
from absence_modeling.preprocessing import build_lesson_frame, prepare_data
from absence_modeling.tasks import build_all_task_datasets, build_chronic_dataset


def test_substantial_absence_threshold(synthetic_project: dict[str, str]) -> None:
    config = load_config(synthetic_project["config"])
    attendance, schedule, calendar = load_inputs(config.data)
    lesson_frame = build_lesson_frame(attendance, schedule, calendar, config)

    row_22 = lesson_frame.iloc[0].copy()
    row_22["scheduled_minutes"] = 45
    row_22["missed_minutes"] = 22
    row_22["miss_share"] = row_22["missed_minutes"] / row_22["scheduled_minutes"]
    assert row_22["miss_share"] < 0.5

    row_23 = lesson_frame.iloc[0].copy()
    row_23["scheduled_minutes"] = 45
    row_23["missed_minutes"] = 23
    row_23["miss_share"] = row_23["missed_minutes"] / row_23["scheduled_minutes"]
    assert row_23["miss_share"] >= 0.5


def test_chronic_target_uses_term_ratio(synthetic_project: dict[str, str]) -> None:
    config = load_config(synthetic_project["config"])
    attendance, schedule, calendar = load_inputs(config.data)
    prepared = prepare_data(attendance, schedule, calendar, config)
    chronic = build_chronic_dataset(prepared, 0.20).frame
    assert "target_chronic_20" in chronic.columns
    assert set(chronic["target_chronic_20"].unique()).issubset({0, 1})
    assert "school_id" not in build_chronic_dataset(prepared, 0.20).features.categorical_features
    assert "class_id" not in build_chronic_dataset(prepared, 0.20).features.categorical_features
    assert "class_other_ratio_today" not in build_chronic_dataset(prepared, 0.20).features.numeric_features
    assert "school_other_ratio_today" not in build_chronic_dataset(prepared, 0.20).features.numeric_features
    assert "has_prior_year_history" in chronic.columns


def test_all_models_use_stable_feature_sets(synthetic_project: dict[str, str]) -> None:
    config = load_config(synthetic_project["config"])
    attendance, schedule, calendar = load_inputs(config.data)
    prepared = prepare_data(attendance, schedule, calendar, config)
    datasets = build_all_task_datasets(prepared, config)

    banned_numeric = {
        "class_other_ratio_today",
        "school_other_ratio_today",
        "class_other_ratio_roll5",
        "school_other_ratio_roll5",
    }
    banned_categorical = {"school_id", "class_id"}

    for dataset in datasets.values():
        assert banned_numeric.isdisjoint(dataset.features.numeric_features)
        assert banned_categorical.isdisjoint(dataset.features.categorical_features)

    assert {
        "calendar_days_since_previous_school_day",
        "calendar_days_until_next_school_day",
        "after_long_break",
        "future_school_days_available",
        "future_total_scheduled_minutes",
        "next_school_day_gap_days",
        "next_school_day_after_long_break",
    }.issubset(set(datasets["short_horizon"].features.numeric_features))

    assert {
        "calendar_days_since_previous_school_day",
        "calendar_days_until_next_school_day",
        "after_long_break",
        "calendar_days_until_target",
        "target_days_since_previous_school_day",
        "target_after_long_break",
    }.issubset(set(datasets["lesson"].features.numeric_features))


def test_run_pipeline_and_score_class(synthetic_project: dict[str, str]) -> None:
    records = run_pipeline(synthetic_project["config"])
    assert any(record["task"] == "short_horizon" and record["split"] == "validation" for record in records)
    assert any(record["task"] == "lesson" and record["split"] == "test" for record in records)

    outputs = score_class(
        synthetic_project["config"],
        as_of_date="2025-08-27",
        school_id="SCHOOL_1",
        class_id="CLASS_7A",
    )
    short_horizon = pd.read_parquet(outputs["short_horizon_predictions"])
    chronic = pd.read_parquet(outputs["chronic_predictions"])
    lesson = pd.read_parquet(outputs["lesson_predictions"])
    lesson_aggregate = pd.read_parquet(outputs["lesson_expected_absence"])

    assert not short_horizon.empty
    assert not chronic.empty
    assert not lesson.empty
    assert not lesson_aggregate.empty
    assert "short_horizon_absence_prob" in short_horizon.columns
    assert {"risk_10pct_term_end", "risk_20pct_term_end"}.issubset(chronic.columns)
    assert "lesson_absence_prob" in lesson.columns
    assert "lesson_expected_absence" in lesson_aggregate.columns
