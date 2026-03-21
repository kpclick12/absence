from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AppConfig
from .utils import academic_year_from_date, grade_to_stage, previous_academic_year, safe_divide, start_minutes, time_bin_from_minutes


JOIN_KEYS = ["student_id", "school_id", "class_id", "grade", "date", "lesson_id"]


@dataclass(slots=True)
class PreparedData:
    lesson_frame: pd.DataFrame
    student_day: pd.DataFrame
    schedule_day: pd.DataFrame
    calendar: pd.DataFrame


def _parse_tables(
    attendance: pd.DataFrame,
    schedule: pd.DataFrame,
    calendar: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attendance = attendance.copy()
    schedule = schedule.copy()
    calendar = calendar.copy()

    for frame in (attendance, schedule, calendar):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    attendance["record_created_at"] = pd.to_datetime(attendance["record_created_at"])
    attendance["record_updated_at"] = pd.to_datetime(attendance["record_updated_at"])
    attendance["scheduled_start_at"] = pd.to_datetime(attendance["scheduled_start_at"], errors="coerce")
    schedule["scheduled_start_at"] = pd.to_datetime(schedule["scheduled_start_at"], errors="coerce")

    numeric_columns = ["scheduled_minutes", "attended_minutes", "missed_minutes"]
    for column in numeric_columns:
        attendance[column] = pd.to_numeric(attendance[column], errors="coerce")
    schedule["scheduled_minutes"] = pd.to_numeric(schedule["scheduled_minutes"], errors="coerce")

    calendar["is_instructional_day"] = calendar["is_instructional_day"].astype(bool)

    attendance["grade"] = attendance["grade"].astype(str)
    schedule["grade"] = schedule["grade"].astype(str)
    calendar["term_id"] = calendar["term_id"].astype(str)
    attendance["term_id"] = attendance["term_id"].astype(str)

    attendance = attendance.sort_values(JOIN_KEYS + ["record_updated_at"]).drop_duplicates(JOIN_KEYS, keep="last")
    schedule = schedule.sort_values(JOIN_KEYS).drop_duplicates(JOIN_KEYS, keep="last")
    calendar = calendar.sort_values(["school_id", "date"]).drop_duplicates(["school_id", "date"], keep="last")

    attendance["academic_year"] = academic_year_from_date(attendance["date"], config.project.school_year_start_month)
    schedule["academic_year"] = academic_year_from_date(schedule["date"], config.project.school_year_start_month)
    calendar["academic_year"] = academic_year_from_date(calendar["date"], config.project.school_year_start_month)

    return attendance, schedule, calendar


def build_lesson_frame(
    attendance: pd.DataFrame,
    schedule: pd.DataFrame,
    calendar: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    attendance, schedule, calendar = _parse_tables(attendance, schedule, calendar, config)

    calendar_subset = calendar[["school_id", "date", "term_id", "is_instructional_day", "school_day_index", "academic_year"]]
    lesson = schedule.merge(calendar_subset, on=["school_id", "date"], how="left", suffixes=("", "_calendar"))
    lesson = lesson.merge(
        attendance[
            JOIN_KEYS
            + [
                "term_id",
                "subject",
                "scheduled_start_at",
                "scheduled_minutes",
                "attended_minutes",
                "missed_minutes",
                "absence_validity",
                "record_created_at",
                "record_updated_at",
            ]
        ].rename(
            columns={
                "term_id": "attendance_term_id",
                "subject": "attendance_subject",
                "scheduled_start_at": "attendance_scheduled_start_at",
                "scheduled_minutes": "attendance_scheduled_minutes",
            }
        ),
        on=JOIN_KEYS,
        how="left",
    )

    lesson["term_id"] = lesson["term_id"].fillna(lesson["attendance_term_id"])
    lesson["subject"] = lesson["subject"].fillna(lesson["attendance_subject"])
    lesson["scheduled_start_at"] = lesson["scheduled_start_at"].fillna(lesson["attendance_scheduled_start_at"])
    lesson["scheduled_minutes"] = lesson["scheduled_minutes"].fillna(lesson["attendance_scheduled_minutes"])

    lesson["has_attendance_record"] = lesson["record_updated_at"].notna()
    lesson["scheduled_minutes"] = pd.to_numeric(lesson["scheduled_minutes"], errors="coerce")
    lesson["attended_minutes"] = pd.to_numeric(lesson["attended_minutes"], errors="coerce")
    lesson["missed_minutes"] = pd.to_numeric(lesson["missed_minutes"], errors="coerce")
    inferred_missed = lesson["scheduled_minutes"] - lesson["attended_minutes"].fillna(0)
    lesson["missed_minutes"] = np.where(
        lesson["has_attendance_record"],
        lesson["missed_minutes"].fillna(inferred_missed.clip(lower=0)),
        np.nan,
    )
    lesson["attended_minutes"] = np.where(
        lesson["has_attendance_record"],
        lesson["attended_minutes"].fillna((lesson["scheduled_minutes"] - lesson["missed_minutes"]).clip(lower=0)),
        np.nan,
    )
    lesson["lesson_start_minutes"] = lesson["scheduled_start_at"].map(start_minutes)
    lesson["lesson_time_bin"] = lesson["lesson_start_minutes"].map(time_bin_from_minutes)
    lesson["stage"] = lesson["grade"].map(grade_to_stage)
    lesson["weekday"] = lesson["date"].dt.dayofweek
    lesson["academic_year"] = lesson["academic_year"].fillna(
        academic_year_from_date(lesson["date"], config.project.school_year_start_month)
    )
    lesson["is_instructional_day"] = lesson["is_instructional_day"].fillna(True)
    lesson = lesson[lesson["is_instructional_day"]].copy()

    lesson["miss_share"] = safe_divide(lesson["missed_minutes"], lesson["scheduled_minutes"])
    lesson["substantial_miss"] = lesson["miss_share"] >= config.modeling.substantial_absence_threshold
    lesson["partial_miss"] = (lesson["missed_minutes"].fillna(0) > 0) & (~lesson["substantial_miss"].fillna(False))
    lesson["is_giltig"] = lesson["absence_validity"].fillna("").str.lower().eq("giltig")
    lesson["is_ogiltig"] = lesson["absence_validity"].fillna("").str.lower().eq("ogiltig")
    lesson["registration_missing"] = ~lesson["has_attendance_record"]

    lesson = lesson.sort_values(["student_id", "date", "lesson_start_minutes", "lesson_id"]).reset_index(drop=True)
    return lesson


def build_schedule_day(lesson_frame: pd.DataFrame) -> pd.DataFrame:
    schedule_day = (
        lesson_frame.groupby(
            ["student_id", "school_id", "class_id", "grade", "stage", "date", "term_id", "academic_year", "school_day_index"],
            as_index=False,
        )
        .agg(
            total_scheduled_lessons=("lesson_id", "size"),
            total_scheduled_minutes=("scheduled_minutes", "sum"),
            first_lesson_start=("lesson_start_minutes", "min"),
            last_lesson_start=("lesson_start_minutes", "max"),
            subject_count=("subject", "nunique"),
            morning_lessons=("lesson_start_minutes", lambda values: int(np.sum(np.asarray(values) < 12 * 60))),
            afternoon_lessons=("lesson_start_minutes", lambda values: int(np.sum(np.asarray(values) >= 12 * 60))),
        )
        .sort_values(["student_id", "date"])
        .reset_index(drop=True)
    )
    schedule_day["calendar_days_since_previous_school_day"] = (
        schedule_day.groupby("student_id")["date"].diff().dt.days
    )
    schedule_day["calendar_days_until_next_school_day"] = (
        schedule_day.groupby("student_id")["date"].shift(-1) - schedule_day["date"]
    ).dt.days
    schedule_day["after_long_break"] = (schedule_day["calendar_days_since_previous_school_day"].fillna(1) >= 4).astype(int)
    return schedule_day


def _rolling_ratio(frame: pd.DataFrame, numerator: str, denominator: str, window: int, output: str) -> None:
    grouped_numerator = frame.groupby("student_id")[numerator]
    grouped_denominator = frame.groupby("student_id")[denominator]
    numerator_sum = grouped_numerator.rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)
    denominator_sum = grouped_denominator.rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)
    frame[output] = safe_divide(numerator_sum, denominator_sum)


def _rolling_mean(frame: pd.DataFrame, column: str, window: int, output: str) -> None:
    frame[output] = (
        frame.groupby("student_id")[column].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    )


def _streak(values: pd.Series) -> list[int]:
    streak = 0
    output: list[int] = []
    for item in values.fillna(False):
        if bool(item):
            streak += 1
        else:
            streak = 0
        output.append(streak)
    return output


def _days_since(values: pd.Series) -> list[float]:
    last_index: int | None = None
    output: list[float] = []
    for index, item in enumerate(values.fillna(False)):
        if bool(item):
            last_index = index
            output.append(0.0)
        elif last_index is None:
            output.append(np.nan)
        else:
            output.append(float(index - last_index))
    return output


def _attach_prior_year_features(student_day: pd.DataFrame) -> pd.DataFrame:
    year_summary = (
        student_day.groupby(["student_id", "academic_year"], as_index=False)
        .agg(
            prior_year_recorded_minutes=("daily_recorded_minutes", "sum"),
            prior_year_missed_minutes=("daily_missed_minutes", "sum"),
            prior_year_giltig_missed=("daily_giltig_missed_minutes", "sum"),
            prior_year_ogiltig_missed=("daily_ogiltig_missed_minutes", "sum"),
            prior_year_substantial_days=("daily_any_substantial_miss", "sum"),
            prior_year_days=("date", "nunique"),
        )
        .sort_values(["student_id", "academic_year"])
    )
    year_summary["prior_year_missed_ratio"] = safe_divide(
        year_summary["prior_year_missed_minutes"], year_summary["prior_year_recorded_minutes"]
    )
    year_summary["prior_year_giltig_ratio"] = safe_divide(
        year_summary["prior_year_giltig_missed"], year_summary["prior_year_recorded_minutes"]
    )
    year_summary["prior_year_ogiltig_ratio"] = safe_divide(
        year_summary["prior_year_ogiltig_missed"], year_summary["prior_year_recorded_minutes"]
    )
    year_summary["prior_year_substantial_day_rate"] = safe_divide(
        year_summary["prior_year_substantial_days"], year_summary["prior_year_days"]
    )

    student_day = student_day.copy()
    student_day["previous_academic_year"] = student_day["academic_year"].map(previous_academic_year)
    prior_columns = [
        "prior_year_missed_ratio",
        "prior_year_giltig_ratio",
        "prior_year_ogiltig_ratio",
        "prior_year_substantial_day_rate",
    ]
    student_day = student_day.merge(
        year_summary[["student_id", "academic_year"] + prior_columns].rename(columns={"academic_year": "previous_academic_year"}),
        on=["student_id", "previous_academic_year"],
        how="left",
    )
    student_day["has_prior_year_history"] = student_day["prior_year_missed_ratio"].notna().astype(int)
    return student_day


def build_student_day(lesson_frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    recorded = lesson_frame[lesson_frame["has_attendance_record"]].copy()

    daily = (
        recorded.groupby(
            ["student_id", "school_id", "class_id", "grade", "stage", "date", "term_id", "academic_year"],
            as_index=False,
        )
        .agg(
            daily_recorded_lessons=("lesson_id", "size"),
            daily_recorded_minutes=("scheduled_minutes", "sum"),
            daily_missed_minutes=("missed_minutes", "sum"),
            daily_substantial_miss_lessons=("substantial_miss", "sum"),
            daily_partial_miss_lessons=("partial_miss", "sum"),
            daily_any_substantial_miss=("substantial_miss", "max"),
        )
        .sort_values(["student_id", "date"])
        .reset_index(drop=True)
    )

    giltig = (
        recorded[recorded["is_giltig"]]
        .groupby(["student_id", "date"], as_index=False)["missed_minutes"]
        .sum()
        .rename(columns={"missed_minutes": "daily_giltig_missed_minutes"})
    )
    ogiltig = (
        recorded[recorded["is_ogiltig"]]
        .groupby(["student_id", "date"], as_index=False)["missed_minutes"]
        .sum()
        .rename(columns={"missed_minutes": "daily_ogiltig_missed_minutes"})
    )

    student_day = build_schedule_day(lesson_frame).merge(daily, on=["student_id", "school_id", "class_id", "grade", "stage", "date", "term_id", "academic_year"], how="left")
    student_day = student_day.merge(giltig, on=["student_id", "date"], how="left")
    student_day = student_day.merge(ogiltig, on=["student_id", "date"], how="left")
    student_day["daily_recorded_lessons"] = student_day["daily_recorded_lessons"].fillna(0)
    student_day["daily_recorded_minutes"] = student_day["daily_recorded_minutes"].fillna(0)
    student_day["daily_missed_minutes"] = student_day["daily_missed_minutes"].fillna(0)
    student_day["daily_giltig_missed_minutes"] = student_day["daily_giltig_missed_minutes"].fillna(0)
    student_day["daily_ogiltig_missed_minutes"] = student_day["daily_ogiltig_missed_minutes"].fillna(0)
    student_day["daily_substantial_miss_lessons"] = student_day["daily_substantial_miss_lessons"].fillna(0)
    student_day["daily_partial_miss_lessons"] = student_day["daily_partial_miss_lessons"].fillna(0)
    student_day["daily_any_substantial_miss"] = student_day["daily_any_substantial_miss"].fillna(False).astype(bool)
    student_day["registration_complete"] = student_day["daily_recorded_lessons"] == student_day["total_scheduled_lessons"]
    student_day["daily_missed_ratio"] = safe_divide(student_day["daily_missed_minutes"], student_day["daily_recorded_minutes"])
    student_day["daily_giltig_ratio"] = safe_divide(student_day["daily_giltig_missed_minutes"], student_day["daily_recorded_minutes"])
    student_day["daily_ogiltig_ratio"] = safe_divide(student_day["daily_ogiltig_missed_minutes"], student_day["daily_recorded_minutes"])
    student_day["daily_partial_lesson_ratio"] = safe_divide(
        student_day["daily_partial_miss_lessons"], student_day["daily_recorded_lessons"]
    )

    student_day = student_day.sort_values(["student_id", "date"]).reset_index(drop=True)
    for window in (1, 3, 5, 10, 20):
        _rolling_ratio(student_day, "daily_missed_minutes", "daily_recorded_minutes", window, f"roll_{window}d_missed_ratio")
        _rolling_ratio(student_day, "daily_giltig_missed_minutes", "daily_recorded_minutes", window, f"roll_{window}d_giltig_ratio")
        _rolling_ratio(student_day, "daily_ogiltig_missed_minutes", "daily_recorded_minutes", window, f"roll_{window}d_ogiltig_ratio")
        _rolling_mean(student_day, "daily_partial_lesson_ratio", window, f"roll_{window}d_partial_ratio")
        _rolling_mean(
            student_day,
            "daily_any_substantial_miss",
            window,
            f"roll_{window}d_substantial_day_rate",
        )

    student_day["substantial_streak_days"] = (
        student_day.groupby("student_id")["daily_any_substantial_miss"].transform(_streak).astype(int)
    )
    student_day["ogiltig_streak_days"] = (
        student_day.groupby("student_id")["daily_ogiltig_missed_minutes"].transform(lambda values: _streak(values > 0)).astype(int)
    )
    student_day["school_days_since_substantial_miss"] = (
        student_day.groupby("student_id")["daily_any_substantial_miss"].transform(_days_since)
    )
    student_day["school_days_since_ogiltig_miss"] = (
        student_day.groupby("student_id")["daily_ogiltig_missed_minutes"].transform(lambda values: _days_since(values > 0))
    )

    student_day["term_cumulative_missed_minutes"] = (
        student_day.groupby(["student_id", "term_id"])["daily_missed_minutes"].cumsum()
    )
    student_day["term_cumulative_recorded_minutes"] = (
        student_day.groupby(["student_id", "term_id"])["daily_recorded_minutes"].cumsum()
    )
    student_day["term_cumulative_giltig_missed_minutes"] = (
        student_day.groupby(["student_id", "term_id"])["daily_giltig_missed_minutes"].cumsum()
    )
    student_day["term_cumulative_ogiltig_missed_minutes"] = (
        student_day.groupby(["student_id", "term_id"])["daily_ogiltig_missed_minutes"].cumsum()
    )
    student_day["term_cumulative_missed_ratio"] = safe_divide(
        student_day["term_cumulative_missed_minutes"], student_day["term_cumulative_recorded_minutes"]
    )
    student_day["term_cumulative_giltig_ratio"] = safe_divide(
        student_day["term_cumulative_giltig_missed_minutes"], student_day["term_cumulative_recorded_minutes"]
    )
    student_day["term_cumulative_ogiltig_ratio"] = safe_divide(
        student_day["term_cumulative_ogiltig_missed_minutes"], student_day["term_cumulative_recorded_minutes"]
    )
    student_day["trend_5_vs_20_missed_ratio"] = student_day["roll_5d_missed_ratio"] - student_day["roll_20d_missed_ratio"]

    calendar_term = (
        calendar.groupby(["school_id", "term_id"], as_index=False)["school_day_index"]
        .max()
        .rename(columns={"school_day_index": "term_total_school_days"})
    )
    student_day = student_day.merge(calendar_term, on=["school_id", "term_id"], how="left")
    student_day["term_days_remaining"] = student_day["term_total_school_days"] - student_day["school_day_index"]
    student_day["term_progress_ratio"] = safe_divide(student_day["school_day_index"], student_day["term_total_school_days"])

    student_day = _attach_prior_year_features(student_day)
    return student_day.sort_values(["student_id", "date"]).reset_index(drop=True)


def prepare_data(
    attendance: pd.DataFrame,
    schedule: pd.DataFrame,
    calendar: pd.DataFrame,
    config: AppConfig,
) -> PreparedData:
    lesson_frame = build_lesson_frame(attendance, schedule, calendar, config)
    prepared_calendar = calendar.copy()
    prepared_calendar["date"] = pd.to_datetime(prepared_calendar["date"]).dt.normalize()
    prepared_calendar["term_id"] = prepared_calendar["term_id"].astype(str)
    student_day = build_student_day(lesson_frame, prepared_calendar)
    schedule_day = build_schedule_day(lesson_frame)
    return PreparedData(lesson_frame=lesson_frame, student_day=student_day, schedule_day=schedule_day, calendar=prepared_calendar)
