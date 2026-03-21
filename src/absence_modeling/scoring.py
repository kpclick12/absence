from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import AppConfig
from .preprocessing import PreparedData
from .tasks import (
    CALENDAR_GAP_FEATURES,
    CHRONIC_NUMERIC_FEATURES,
    LESSON_CATEGORICAL_FEATURES,
    LESSON_NUMERIC_FEATURES,
    SHORT_HORIZON_CATEGORICAL_FEATURES,
    SHORT_HORIZON_NUMERIC_FEATURES,
    STUDENT_DAY_CATEGORICAL_FEATURES,
    STUDENT_DAY_NUMERIC_FEATURES,
    _add_common_columns,
    _build_short_horizon_frame,
    _build_slot_history,
    _merge_history,
)


def _select_unique_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ordered: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return frame[ordered].copy()


@dataclass(slots=True)
class ClassScoreFrames:
    short_horizon: pd.DataFrame
    chronic: pd.DataFrame
    lesson: pd.DataFrame


def _cohort_filter(frame: pd.DataFrame, school_id: str, class_id: str) -> pd.DataFrame:
    return frame[(frame["school_id"].astype(str) == str(school_id)) & (frame["class_id"].astype(str) == str(class_id))].copy()


def build_class_scoring_frames(
    prepared: PreparedData,
    config: AppConfig,
    as_of_date: str,
    school_id: str,
    class_id: str,
) -> ClassScoreFrames:
    scoring_date = pd.Timestamp(as_of_date).normalize()
    student_day = _add_common_columns(prepared.student_day).sort_values(["student_id", "date"]).reset_index(drop=True)
    student_day["student_day_pos"] = student_day.groupby("student_id").cumcount()

    current_day = _cohort_filter(student_day[student_day["date"] == scoring_date], school_id, class_id)
    if current_day.empty:
        raise ValueError(f"No scored students found for school={school_id} class={class_id} on {scoring_date.date()}")

    short_horizon_source = _build_short_horizon_frame(prepared.student_day, config.modeling.short_horizon_school_days)
    short_horizon = _cohort_filter(short_horizon_source[short_horizon_source["date"] == scoring_date], school_id, class_id)

    chronic = current_day.copy()
    chronic["gap_to_10pct"] = 0.10 - chronic["term_cumulative_missed_ratio"]
    chronic["gap_to_20pct"] = 0.20 - chronic["term_cumulative_missed_ratio"]

    current_index = current_day[["student_id", "student_day_pos"]].rename(columns={"student_day_pos": "current_pos"})
    future_days = student_day.merge(current_index, on="student_id", how="inner")
    future_days["horizon_school_days"] = future_days["student_day_pos"] - future_days["current_pos"]
    future_days = future_days[
        (future_days["horizon_school_days"] >= 1)
        & (future_days["horizon_school_days"] <= config.modeling.lesson_horizon_school_days)
    ][["student_id", "date", "horizon_school_days"]].rename(columns={"date": "target_date"})

    lesson_frame = prepared.lesson_frame.copy()
    lesson_frame["weekday_name"] = lesson_frame["date"].dt.day_name().str.lower()
    lesson_scoring = lesson_frame.rename(columns={"date": "target_date"}).merge(
        future_days,
        on=["student_id", "target_date"],
        how="inner",
    )
    lesson_scoring = lesson_scoring.merge(
        current_day[["student_id", "date"] + STUDENT_DAY_NUMERIC_FEATURES + CALENDAR_GAP_FEATURES + ["weekday_name"]].rename(columns={"date": "score_date"}),
        on="student_id",
        how="left",
    )
    lesson_scoring["target_weekday_name"] = lesson_scoring["target_date"].dt.day_name().str.lower()
    lesson_scoring["target_first_lesson_flag"] = (
        lesson_scoring.groupby(["student_id", "target_date"])["lesson_start_minutes"].transform("min")
        == lesson_scoring["lesson_start_minutes"]
    ).map({True: "yes", False: "no"})
    lesson_scoring["target_last_lesson_flag"] = (
        lesson_scoring.groupby(["student_id", "target_date"])["lesson_start_minutes"].transform("max")
        == lesson_scoring["lesson_start_minutes"]
    ).map({True: "yes", False: "no"})
    lesson_scoring["target_scheduled_minutes"] = lesson_scoring["scheduled_minutes"]
    lesson_scoring["target_lesson_start_minutes"] = lesson_scoring["lesson_start_minutes"]
    lesson_scoring["target_subject"] = lesson_scoring["subject"]
    lesson_scoring["target_lesson_time_bin"] = lesson_scoring["lesson_time_bin"]
    lesson_scoring = lesson_scoring.drop(columns=["subject", "lesson_time_bin"], errors="ignore")
    lesson_scoring = lesson_scoring.rename(columns={"weekday_name": "score_weekday_name"})

    target_day_schedule = prepared.schedule_day.rename(
        columns={
            "date": "target_date",
            "total_scheduled_lessons": "target_day_total_scheduled_lessons",
            "total_scheduled_minutes": "target_day_total_scheduled_minutes",
            "subject_count": "target_day_subject_count",
            "calendar_days_since_previous_school_day": "target_days_since_previous_school_day",
            "after_long_break": "target_after_long_break",
        }
    )[
        [
            "student_id",
            "target_date",
            "target_day_total_scheduled_lessons",
            "target_day_total_scheduled_minutes",
            "target_day_subject_count",
            "target_days_since_previous_school_day",
            "target_after_long_break",
        ]
    ]
    lesson_scoring = lesson_scoring.merge(target_day_schedule, on=["student_id", "target_date"], how="left")
    lesson_scoring["calendar_days_until_target"] = (lesson_scoring["target_date"] - lesson_scoring["score_date"]).dt.days

    subject_history = _build_slot_history(lesson_frame, ["student_id", "subject"], "hist_subject_lesson_count", "hist_subject_substantial_rate")
    weekday_history = _build_slot_history(lesson_frame, ["student_id", "weekday_name"], "hist_weekday_lesson_count", "hist_weekday_substantial_rate")
    timebin_history = _build_slot_history(lesson_frame, ["student_id", "lesson_time_bin"], "hist_timebin_lesson_count", "hist_timebin_substantial_rate")
    subject_slot_history = _build_slot_history(
        lesson_frame,
        ["student_id", "subject", "weekday_name", "lesson_time_bin"],
        "hist_subject_weekday_timebin_lesson_count",
        "hist_subject_weekday_timebin_substantial_rate",
    )
    lesson_scoring = _merge_history(
        lesson_scoring.rename(columns={"target_subject": "subject"}),
        subject_history,
        ["student_id", "subject"],
        "hist_subject_lesson_count",
        "hist_subject_substantial_rate",
    ).rename(columns={"subject": "target_subject"})
    lesson_scoring = _merge_history(
        lesson_scoring.rename(columns={"target_weekday_name": "weekday_name"}),
        weekday_history,
        ["student_id", "weekday_name"],
        "hist_weekday_lesson_count",
        "hist_weekday_substantial_rate",
    ).rename(columns={"weekday_name": "target_weekday_name"})
    lesson_scoring = _merge_history(
        lesson_scoring.rename(columns={"target_lesson_time_bin": "lesson_time_bin"}),
        timebin_history,
        ["student_id", "lesson_time_bin"],
        "hist_timebin_lesson_count",
        "hist_timebin_substantial_rate",
    ).rename(columns={"lesson_time_bin": "target_lesson_time_bin"})
    lesson_scoring = _merge_history(
        lesson_scoring.rename(
            columns={
                "target_subject": "subject",
                "target_weekday_name": "weekday_name",
                "target_lesson_time_bin": "lesson_time_bin",
            }
        ),
        subject_slot_history,
        ["student_id", "subject", "weekday_name", "lesson_time_bin"],
        "hist_subject_weekday_timebin_lesson_count",
        "hist_subject_weekday_timebin_substantial_rate",
    ).rename(
        columns={
            "subject": "target_subject",
            "weekday_name": "target_weekday_name",
            "lesson_time_bin": "target_lesson_time_bin",
        }
    )
    lesson_scoring["school_day_index"] = lesson_scoring.get("school_day_index_y", lesson_scoring.get("school_day_index_x"))
    lesson_scoring["weekday_name"] = lesson_scoring.get(
        "weekday_name_y", lesson_scoring.get("score_weekday_name", lesson_scoring.get("weekday_name_x"))
    )
    lesson_scoring = lesson_scoring.drop(
        columns=[
            "school_day_index_x",
            "school_day_index_y",
            "weekday_name_x",
            "weekday_name_y",
            "score_weekday_name",
        ],
        errors="ignore",
    )

    short_horizon = _select_unique_columns(
        short_horizon,
        ["date", "student_id", "school_id", "class_id", "grade", "stage"]
        + SHORT_HORIZON_NUMERIC_FEATURES
        + SHORT_HORIZON_CATEGORICAL_FEATURES,
    )
    chronic = _select_unique_columns(
        chronic,
        ["date", "student_id", "school_id", "class_id", "grade", "stage"]
        + CHRONIC_NUMERIC_FEATURES
        + STUDENT_DAY_CATEGORICAL_FEATURES,
    )
    lesson_scoring = _select_unique_columns(
        lesson_scoring,
        [
            "score_date",
            "target_date",
            "student_id",
            "school_id",
            "class_id",
            "grade",
            "stage",
            "lesson_id",
        ]
        + LESSON_NUMERIC_FEATURES
        + LESSON_CATEGORICAL_FEATURES,
    )
    return ClassScoreFrames(short_horizon=short_horizon, chronic=chronic, lesson=lesson_scoring)
