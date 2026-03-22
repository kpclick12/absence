from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import AppConfig
from .preprocessing import PreparedData
from .utils import academic_year_from_date, safe_divide
from .tasks import (
    CALENDAR_GAP_FEATURES,
    CHRONIC_LASYAR_NUMERIC_FEATURES,
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
    chronic_lasyar: pd.DataFrame
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

    # Läsår (school-year) chronic scoring frame
    all_student_day = _add_common_columns(student_day)
    current_year = current_day["academic_year"].iloc[0] if not current_day.empty else None
    year_rows = all_student_day[
        (all_student_day["student_id"].isin(current_day["student_id"]))
        & (all_student_day["academic_year"] == current_year)
        & (all_student_day["date"] <= scoring_date)
    ].copy()

    year_rows = year_rows.sort_values(["student_id", "date"])
    year_cum_missed = year_rows.groupby("student_id")["daily_missed_minutes"].cumsum()
    year_cum_recorded = year_rows.groupby("student_id")["daily_recorded_minutes"].cumsum()
    year_rows["year_cumulative_missed_ratio"] = safe_divide(year_cum_missed, year_cum_recorded)
    year_rows["gap_to_year_10pct"] = 0.10 - year_rows["year_cumulative_missed_ratio"]
    year_rows["gap_to_year_20pct"] = 0.20 - year_rows["year_cumulative_missed_ratio"]
    year_rows["is_term2"] = (
        year_rows["date"].dt.month < config.project.school_year_start_month
    ).astype(float)

    term1_finals = (
        year_rows[year_rows["is_term2"] == 0]
        .sort_values("date")
        .groupby("student_id", as_index=False)
        .last()[["student_id", "term_cumulative_missed_ratio", "term_cumulative_giltig_ratio", "term_cumulative_ogiltig_ratio"]]
        .rename(columns={
            "term_cumulative_missed_ratio": "term1_actual_missed_ratio",
            "term_cumulative_giltig_ratio": "term1_actual_giltig_ratio",
            "term_cumulative_ogiltig_ratio": "term1_actual_ogiltig_ratio",
        })
    )
    year_rows = year_rows.merge(term1_finals, on="student_id", how="left")
    for col in ("term1_actual_missed_ratio", "term1_actual_giltig_ratio", "term1_actual_ogiltig_ratio"):
        year_rows.loc[year_rows["is_term2"] == 0, col] = 0.0
        year_rows[col] = year_rows[col].fillna(0.0)

    year_rows["year_school_days_elapsed"] = year_rows.groupby("student_id").cumcount() + 1
    scoring_calendar = prepared.calendar.copy()
    scoring_calendar["academic_year"] = academic_year_from_date(
        pd.to_datetime(scoring_calendar["date"]), config.project.school_year_start_month
    )
    year_calendar_totals = (
        scoring_calendar[scoring_calendar["is_instructional_day"]]
        .groupby(["school_id", "academic_year"], as_index=False)
        .agg(year_total_school_days=("date", "nunique"))
    )
    year_rows = year_rows.merge(year_calendar_totals, on=["school_id", "academic_year"], how="left")
    year_rows["year_progress_ratio"] = safe_divide(
        year_rows["year_school_days_elapsed"], year_rows["year_total_school_days"]
    )
    year_rows["year_days_remaining"] = year_rows["year_total_school_days"] - year_rows["year_school_days_elapsed"]
    year_rows["gap_to_10pct"] = 0.10 - year_rows["term_cumulative_missed_ratio"]
    year_rows["gap_to_20pct"] = 0.20 - year_rows["term_cumulative_missed_ratio"]

    # Take only the scoring-date row per student
    chronic_lasyar = year_rows[year_rows["date"] == scoring_date].copy()
    chronic_lasyar = _cohort_filter(chronic_lasyar, school_id, class_id)
    chronic_lasyar = _select_unique_columns(
        chronic_lasyar,
        ["date", "student_id", "school_id", "class_id", "grade", "stage"]
        + CHRONIC_LASYAR_NUMERIC_FEATURES
        + STUDENT_DAY_CATEGORICAL_FEATURES,
    )

    return ClassScoreFrames(short_horizon=short_horizon, chronic=chronic, chronic_lasyar=chronic_lasyar, lesson=lesson_scoring)
