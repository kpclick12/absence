from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AppConfig
from .preprocessing import PreparedData
from .utils import FeatureBundle, academic_year_from_date, safe_divide


STUDENT_DAY_NUMERIC_FEATURES = [
    "grade_numeric",
    "school_day_index",
    "term_progress_ratio",
    "term_days_remaining",
    "daily_recorded_lessons",
    "daily_recorded_minutes",
    "daily_missed_minutes",
    "daily_missed_ratio",
    "daily_giltig_ratio",
    "daily_ogiltig_ratio",
    "daily_partial_lesson_ratio",
    "daily_substantial_miss_lessons",
    "roll_1d_missed_ratio",
    "roll_3d_missed_ratio",
    "roll_5d_missed_ratio",
    "roll_10d_missed_ratio",
    "roll_20d_missed_ratio",
    "roll_1d_giltig_ratio",
    "roll_3d_giltig_ratio",
    "roll_5d_giltig_ratio",
    "roll_10d_giltig_ratio",
    "roll_20d_giltig_ratio",
    "roll_1d_ogiltig_ratio",
    "roll_3d_ogiltig_ratio",
    "roll_5d_ogiltig_ratio",
    "roll_10d_ogiltig_ratio",
    "roll_20d_ogiltig_ratio",
    "roll_1d_partial_ratio",
    "roll_3d_partial_ratio",
    "roll_5d_partial_ratio",
    "roll_10d_partial_ratio",
    "roll_20d_partial_ratio",
    "roll_1d_substantial_day_rate",
    "roll_3d_substantial_day_rate",
    "roll_5d_substantial_day_rate",
    "roll_10d_substantial_day_rate",
    "roll_20d_substantial_day_rate",
    "substantial_streak_days",
    "ogiltig_streak_days",
    "school_days_since_substantial_miss",
    "school_days_since_ogiltig_miss",
    "term_cumulative_missed_ratio",
    "term_cumulative_giltig_ratio",
    "term_cumulative_ogiltig_ratio",
    "trend_5_vs_20_missed_ratio",
    "has_prior_year_history",
    "prior_year_missed_ratio",
    "prior_year_giltig_ratio",
    "prior_year_ogiltig_ratio",
    "prior_year_substantial_day_rate",
]

STUDENT_DAY_CATEGORICAL_FEATURES = [
    "stage",
    "weekday_name",
]

CALENDAR_GAP_FEATURES = [
    "calendar_days_since_previous_school_day",
    "calendar_days_until_next_school_day",
    "after_long_break",
]

SHORT_HORIZON_NUMERIC_FEATURES = STUDENT_DAY_NUMERIC_FEATURES + CALENDAR_GAP_FEATURES + [
    "future_school_days_available",
    "future_total_scheduled_lessons",
    "future_total_scheduled_minutes",
    "future_average_daily_minutes",
    "future_total_subject_count",
    "future_morning_lessons",
    "future_afternoon_lessons",
    "next_school_day_total_scheduled_lessons",
    "next_school_day_total_scheduled_minutes",
    "next_school_day_first_lesson_start",
    "next_school_day_last_lesson_start",
    "next_school_day_subject_count",
    "next_school_day_morning_lessons",
    "next_school_day_afternoon_lessons",
    "next_school_day_gap_days",
    "next_school_day_after_long_break",
]

SHORT_HORIZON_CATEGORICAL_FEATURES = STUDENT_DAY_CATEGORICAL_FEATURES + [
    "next_school_day_weekday_name",
]

CHRONIC_NUMERIC_FEATURES = STUDENT_DAY_NUMERIC_FEATURES + [
    "gap_to_10pct",
    "gap_to_20pct",
]

CHRONIC_LASYAR_NUMERIC_FEATURES = STUDENT_DAY_NUMERIC_FEATURES + [
    "gap_to_10pct",
    "gap_to_20pct",
    "year_cumulative_missed_ratio",
    "gap_to_year_10pct",
    "gap_to_year_20pct",
    "is_term2",
    "term1_actual_missed_ratio",
    "term1_actual_giltig_ratio",
    "term1_actual_ogiltig_ratio",
    "year_school_days_elapsed",
    "year_total_school_days",
    "year_progress_ratio",
    "year_days_remaining",
]

LESSON_NUMERIC_FEATURES = STUDENT_DAY_NUMERIC_FEATURES + CALENDAR_GAP_FEATURES + [
    "horizon_school_days",
    "calendar_days_until_target",
    "target_scheduled_minutes",
    "target_lesson_start_minutes",
    "target_day_total_scheduled_lessons",
    "target_day_total_scheduled_minutes",
    "target_day_subject_count",
    "target_days_since_previous_school_day",
    "target_after_long_break",
    "hist_subject_lesson_count",
    "hist_subject_substantial_rate",
    "hist_weekday_lesson_count",
    "hist_weekday_substantial_rate",
    "hist_timebin_lesson_count",
    "hist_timebin_substantial_rate",
    "hist_subject_weekday_timebin_lesson_count",
    "hist_subject_weekday_timebin_substantial_rate",
]

LESSON_CATEGORICAL_FEATURES = STUDENT_DAY_CATEGORICAL_FEATURES + [
    "target_subject",
    "target_weekday_name",
    "target_lesson_time_bin",
    "target_first_lesson_flag",
    "target_last_lesson_flag",
]


@dataclass(slots=True)
class TaskDataset:
    name: str
    frame: pd.DataFrame
    target_column: str
    features: FeatureBundle
    id_columns: list[str]
    primary_group_column: str = "stage"


def _add_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["grade_numeric"] = pd.to_numeric(output["grade"], errors="coerce")
    output["weekday_name"] = pd.to_datetime(output["date"]).dt.day_name().str.lower()
    return output


def _filter_years(frame: pd.DataFrame, years: list[str] | None) -> pd.DataFrame:
    if not years:
        return frame
    return frame[frame["academic_year"].isin(years)].copy()


def _select_unique_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ordered: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return frame[ordered].copy()


def _build_short_horizon_frame(student_day: pd.DataFrame, horizon: int) -> pd.DataFrame:
    student_day = _add_common_columns(student_day)
    student_day = student_day.sort_values(["student_id", "date"]).reset_index(drop=True)

    frame = student_day.copy()
    frame["target_short_horizon"] = 0
    frame["future_horizon_complete"] = True
    frame["future_school_days_available"] = 0
    frame["future_total_scheduled_lessons"] = 0.0
    frame["future_total_scheduled_minutes"] = 0.0
    frame["future_total_subject_count"] = 0.0
    frame["future_morning_lessons"] = 0.0
    frame["future_afternoon_lessons"] = 0.0

    grouped = student_day.groupby("student_id", sort=False)
    future_columns = [
        "date",
        "academic_year",
        "registration_complete",
        "daily_any_substantial_miss",
        "total_scheduled_lessons",
        "total_scheduled_minutes",
        "first_lesson_start",
        "last_lesson_start",
        "subject_count",
        "morning_lessons",
        "afternoon_lessons",
        "calendar_days_since_previous_school_day",
        "after_long_break",
    ]

    for horizon_step in range(1, horizon + 1):
        shifted = grouped[future_columns].shift(-horizon_step)
        same_year = shifted["academic_year"].eq(frame["academic_year"]).fillna(False)
        frame["future_horizon_complete"] &= same_year & shifted["registration_complete"].fillna(False)
        frame["future_school_days_available"] += same_year.astype(int)
        frame["target_short_horizon"] = np.maximum(
            frame["target_short_horizon"],
            (same_year & shifted["daily_any_substantial_miss"].fillna(False)).astype(int),
        )
        frame["future_total_scheduled_lessons"] += shifted["total_scheduled_lessons"].where(same_year, 0).fillna(0.0)
        frame["future_total_scheduled_minutes"] += shifted["total_scheduled_minutes"].where(same_year, 0).fillna(0.0)
        frame["future_total_subject_count"] += shifted["subject_count"].where(same_year, 0).fillna(0.0)
        frame["future_morning_lessons"] += shifted["morning_lessons"].where(same_year, 0).fillna(0.0)
        frame["future_afternoon_lessons"] += shifted["afternoon_lessons"].where(same_year, 0).fillna(0.0)

        if horizon_step == 1:
            frame["next_school_day_total_scheduled_lessons"] = shifted["total_scheduled_lessons"].where(same_year)
            frame["next_school_day_total_scheduled_minutes"] = shifted["total_scheduled_minutes"].where(same_year)
            frame["next_school_day_first_lesson_start"] = shifted["first_lesson_start"].where(same_year)
            frame["next_school_day_last_lesson_start"] = shifted["last_lesson_start"].where(same_year)
            frame["next_school_day_subject_count"] = shifted["subject_count"].where(same_year)
            frame["next_school_day_morning_lessons"] = shifted["morning_lessons"].where(same_year)
            frame["next_school_day_afternoon_lessons"] = shifted["afternoon_lessons"].where(same_year)
            frame["next_school_day_gap_days"] = shifted["calendar_days_since_previous_school_day"].where(same_year)
            frame["next_school_day_after_long_break"] = shifted["after_long_break"].where(same_year)
            frame["next_school_day_weekday_name"] = pd.to_datetime(shifted["date"]).dt.day_name().str.lower().where(same_year)

    frame["future_average_daily_minutes"] = safe_divide(
        frame["future_total_scheduled_minutes"], frame["future_school_days_available"]
    )
    frame["target_short_horizon"] = frame["target_short_horizon"].astype(int)
    return frame


def build_short_horizon_dataset(prepared: PreparedData, config: AppConfig, years: list[str] | None = None) -> TaskDataset:
    student_day = _build_short_horizon_frame(prepared.student_day, config.modeling.short_horizon_school_days)
    frame = student_day[student_day["future_horizon_complete"]].copy()
    frame = _filter_years(frame, years)

    feature_columns = SHORT_HORIZON_NUMERIC_FEATURES + SHORT_HORIZON_CATEGORICAL_FEATURES
    features = FeatureBundle(
        frame=frame,
        numeric_features=SHORT_HORIZON_NUMERIC_FEATURES,
        categorical_features=SHORT_HORIZON_CATEGORICAL_FEATURES,
    )
    return TaskDataset(
        name="short_horizon",
        frame=_select_unique_columns(
            frame,
            [
                "date",
                "student_id",
                "school_id",
                "class_id",
                "grade",
                "stage",
                "academic_year",
                "target_short_horizon",
            ]
            + feature_columns,
        ),
        target_column="target_short_horizon",
        features=features,
        id_columns=["date", "student_id", "school_id", "class_id", "grade", "stage", "academic_year"],
    )


def build_chronic_dataset(
    prepared: PreparedData,
    threshold: float,
    years: list[str] | None = None,
) -> TaskDataset:
    student_day = _add_common_columns(prepared.student_day)
    term_totals = (
        student_day.groupby(["student_id", "term_id"], as_index=False)
        .agg(
            final_term_missed_minutes=("daily_missed_minutes", "sum"),
            final_term_recorded_minutes=("daily_recorded_minutes", "sum"),
        )
        .assign(
            final_term_missed_ratio=lambda frame: safe_divide(
                frame["final_term_missed_minutes"], frame["final_term_recorded_minutes"]
            )
        )
    )
    frame = student_day.merge(term_totals, on=["student_id", "term_id"], how="left")
    frame["gap_to_10pct"] = 0.10 - frame["term_cumulative_missed_ratio"]
    frame["gap_to_20pct"] = 0.20 - frame["term_cumulative_missed_ratio"]
    frame[f"target_chronic_{int(threshold * 100)}"] = (frame["final_term_missed_ratio"] >= threshold).astype(int)
    frame = _filter_years(frame, years)

    target_column = f"target_chronic_{int(threshold * 100)}"
    feature_columns = CHRONIC_NUMERIC_FEATURES + STUDENT_DAY_CATEGORICAL_FEATURES
    features = FeatureBundle(
        frame=frame,
        numeric_features=CHRONIC_NUMERIC_FEATURES,
        categorical_features=STUDENT_DAY_CATEGORICAL_FEATURES,
    )
    return TaskDataset(
        name=f"chronic_{int(threshold * 100)}",
        frame=_select_unique_columns(
            frame,
            ["date", "student_id", "school_id", "class_id", "grade", "stage", "term_id", "academic_year", target_column]
            + feature_columns,
        ),
        target_column=target_column,
        features=features,
        id_columns=["date", "student_id", "school_id", "class_id", "grade", "stage", "term_id", "academic_year"],
    )


def _build_slot_history(
    lesson_frame: pd.DataFrame,
    group_columns: list[str],
    count_column_name: str,
    rate_column_name: str,
) -> pd.DataFrame:
    history = (
        lesson_frame[lesson_frame["has_attendance_record"]]
        .groupby(group_columns + ["date"], as_index=False)
        .agg(lesson_count=("lesson_id", "size"), substantial_count=("substantial_miss", "sum"))
        .sort_values(group_columns + ["date"])
    )
    history["cum_lesson_count"] = history.groupby(group_columns)["lesson_count"].cumsum()
    history["cum_substantial_count"] = history.groupby(group_columns)["substantial_count"].cumsum()
    history[count_column_name] = history["cum_lesson_count"]
    history[rate_column_name] = safe_divide(history["cum_substantial_count"], history["cum_lesson_count"])
    return history[group_columns + ["date", count_column_name, rate_column_name]]


def _merge_history(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    by: list[str],
    count_column_name: str,
    rate_column_name: str,
) -> pd.DataFrame:
    history_groups = {
        key if isinstance(key, tuple) else (key,): subgroup.sort_values("date")
        for key, subgroup in history.groupby(by, dropna=False, sort=False)
    }
    merged_frames: list[pd.DataFrame] = []
    for key, subgroup in frame.groupby(by, dropna=False, sort=False):
        normalized_key = key if isinstance(key, tuple) else (key,)
        left = subgroup.sort_values("score_date").copy()
        right = history_groups.get(normalized_key)
        if right is None:
            left[count_column_name] = 0.0
            left[rate_column_name] = 0.0
            merged_frames.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right.drop(columns=by, errors="ignore").sort_values("date"),
            left_on="score_date",
            right_on="date",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["date"], errors="ignore")
        merged[count_column_name] = merged[count_column_name].fillna(0.0)
        merged[rate_column_name] = merged[rate_column_name].fillna(0.0)
        merged_frames.append(merged)
    if not merged_frames:
        return frame
    return pd.concat(merged_frames, ignore_index=True)


def build_lesson_dataset(prepared: PreparedData, config: AppConfig, years: list[str] | None = None) -> TaskDataset:
    lesson_frame = prepared.lesson_frame.copy()
    lesson_frame["weekday_name"] = lesson_frame["date"].dt.day_name().str.lower()

    score_features = _add_common_columns(prepared.student_day).copy()
    score_features["student_day_pos"] = score_features.groupby("student_id").cumcount()
    score_index = score_features[["student_id", "date", "student_day_pos"]].rename(columns={"date": "score_date"})

    target_lessons = lesson_frame[lesson_frame["has_attendance_record"]].copy()
    target_lessons = target_lessons[target_lessons["registration_missing"].eq(False)].copy()
    target_lessons = _filter_years(target_lessons, years)
    target_lessons["target_weekday_name"] = target_lessons["date"].dt.day_name().str.lower()
    target_lessons["target_first_lesson_flag"] = (
        target_lessons.groupby(["student_id", "date"])["lesson_start_minutes"].transform("min")
        == target_lessons["lesson_start_minutes"]
    ).map({True: "yes", False: "no"})
    target_lessons["target_last_lesson_flag"] = (
        target_lessons.groupby(["student_id", "date"])["lesson_start_minutes"].transform("max")
        == target_lessons["lesson_start_minutes"]
    ).map({True: "yes", False: "no"})
    target_lessons["target_label"] = target_lessons["substantial_miss"].astype(int)
    target_lessons = target_lessons.rename(columns={"date": "target_date"})

    target_day_index = score_features[["student_id", "date", "student_day_pos"]].rename(columns={"date": "target_date"})
    target_lessons = target_lessons.merge(target_day_index, on=["student_id", "target_date"], how="left")

    expanded_frames: list[pd.DataFrame] = []
    for horizon in range(1, config.modeling.lesson_horizon_school_days + 1):
        part = target_lessons.copy()
        part["score_pos"] = part["student_day_pos"] - horizon
        part = part[part["score_pos"] >= 0].copy()
        part["horizon_school_days"] = horizon
        part = part.merge(
            score_index.rename(columns={"student_day_pos": "score_pos"}),
            on=["student_id", "score_pos"],
            how="left",
        )
        expanded_frames.append(part)
    frame = pd.concat(expanded_frames, ignore_index=True) if expanded_frames else pd.DataFrame()
    if frame.empty:
        features = FeatureBundle(frame=frame, numeric_features=LESSON_NUMERIC_FEATURES, categorical_features=LESSON_CATEGORICAL_FEATURES)
        return TaskDataset(name="lesson", frame=frame, target_column="target_label", features=features, id_columns=[])

    score_feature_columns = ["student_id", "score_date"] + STUDENT_DAY_NUMERIC_FEATURES + CALENDAR_GAP_FEATURES + ["weekday_name"]
    score_feature_frame = score_features.rename(columns={"date": "score_date"})[score_feature_columns]
    frame = frame.merge(score_feature_frame, on=["student_id", "score_date"], how="left")

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
    frame = frame.merge(target_day_schedule, on=["student_id", "target_date"], how="left")
    frame["calendar_days_until_target"] = (frame["target_date"] - frame["score_date"]).dt.days
    frame["target_scheduled_minutes"] = frame["scheduled_minutes"]
    frame["target_lesson_start_minutes"] = frame["lesson_start_minutes"]
    frame["target_subject"] = frame["subject"]
    frame["target_lesson_time_bin"] = frame["lesson_time_bin"]
    frame = frame.drop(columns=["subject", "lesson_time_bin"], errors="ignore")
    frame = frame.rename(columns={"weekday_name": "score_weekday_name"})

    subject_history = _build_slot_history(lesson_frame, ["student_id", "subject"], "hist_subject_lesson_count", "hist_subject_substantial_rate")
    weekday_history = _build_slot_history(lesson_frame, ["student_id", "weekday_name"], "hist_weekday_lesson_count", "hist_weekday_substantial_rate")
    timebin_history = _build_slot_history(lesson_frame, ["student_id", "lesson_time_bin"], "hist_timebin_lesson_count", "hist_timebin_substantial_rate")
    subject_slot_history = _build_slot_history(
        lesson_frame,
        ["student_id", "subject", "weekday_name", "lesson_time_bin"],
        "hist_subject_weekday_timebin_lesson_count",
        "hist_subject_weekday_timebin_substantial_rate",
    )

    frame = _merge_history(
        frame.rename(columns={"target_subject": "subject"}),
        subject_history,
        ["student_id", "subject"],
        "hist_subject_lesson_count",
        "hist_subject_substantial_rate",
    ).rename(columns={"subject": "target_subject"})
    frame = _merge_history(
        frame.rename(columns={"target_weekday_name": "weekday_name"}),
        weekday_history,
        ["student_id", "weekday_name"],
        "hist_weekday_lesson_count",
        "hist_weekday_substantial_rate",
    ).rename(columns={"weekday_name": "target_weekday_name"})
    frame = _merge_history(
        frame.rename(columns={"target_lesson_time_bin": "lesson_time_bin"}),
        timebin_history,
        ["student_id", "lesson_time_bin"],
        "hist_timebin_lesson_count",
        "hist_timebin_substantial_rate",
    ).rename(columns={"lesson_time_bin": "target_lesson_time_bin"})
    frame = _merge_history(
        frame.rename(
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
    frame["school_day_index"] = frame.get("school_day_index_y", frame.get("school_day_index_x"))
    frame["weekday_name"] = frame.get("weekday_name_y", frame.get("score_weekday_name", frame.get("weekday_name_x")))
    frame = frame.drop(
        columns=[
            "school_day_index_x",
            "school_day_index_y",
            "weekday_name_x",
            "weekday_name_y",
            "score_weekday_name",
        ],
        errors="ignore",
    )

    feature_columns = LESSON_NUMERIC_FEATURES + LESSON_CATEGORICAL_FEATURES
    features = FeatureBundle(frame=frame, numeric_features=LESSON_NUMERIC_FEATURES, categorical_features=LESSON_CATEGORICAL_FEATURES)
    return TaskDataset(
        name="lesson",
        frame=_select_unique_columns(
            frame.rename(columns={"target_date": "date"}),
            [
                "score_date",
                "date",
                "student_id",
                "school_id",
                "class_id",
                "grade",
                "stage",
                "lesson_id",
                "academic_year",
                "target_label",
            ]
            + feature_columns,
        ),
        target_column="target_label",
        features=features,
        id_columns=["score_date", "date", "student_id", "school_id", "class_id", "grade", "stage", "lesson_id", "academic_year"],
    )


def build_chronic_lasyar_dataset(
    prepared: PreparedData,
    threshold: float,
    config: AppConfig,
    years: list[str] | None = None,
) -> TaskDataset:
    student_day = _add_common_columns(prepared.student_day)

    # Year-end target: total missed / total recorded across all terms in the academic year
    year_totals = (
        student_day.groupby(["student_id", "academic_year"], as_index=False)
        .agg(
            final_year_missed_minutes=("daily_missed_minutes", "sum"),
            final_year_recorded_minutes=("daily_recorded_minutes", "sum"),
        )
        .assign(
            final_year_missed_ratio=lambda f: safe_divide(
                f["final_year_missed_minutes"], f["final_year_recorded_minutes"]
            )
        )
    )
    frame = student_day.merge(year_totals, on=["student_id", "academic_year"], how="left")
    target_column = f"target_chronic_year_{int(threshold * 100)}"
    frame[target_column] = (frame["final_year_missed_ratio"] >= threshold).astype(int)

    # Term-level gap features (same as chronic model)
    frame["gap_to_10pct"] = 0.10 - frame["term_cumulative_missed_ratio"]
    frame["gap_to_20pct"] = 0.20 - frame["term_cumulative_missed_ratio"]

    # Year-cumulative missed ratio (resets only at academic year boundary, not at each term)
    frame = frame.sort_values(["student_id", "academic_year", "date"])
    year_cum_missed = frame.groupby(["student_id", "academic_year"])["daily_missed_minutes"].cumsum()
    year_cum_recorded = frame.groupby(["student_id", "academic_year"])["daily_recorded_minutes"].cumsum()
    frame["year_cumulative_missed_ratio"] = safe_divide(year_cum_missed, year_cum_recorded)
    frame["gap_to_year_10pct"] = 0.10 - frame["year_cumulative_missed_ratio"]
    frame["gap_to_year_20pct"] = 0.20 - frame["year_cumulative_missed_ratio"]

    # is_term2: spring term rows (Jan-Jul when school year starts in Aug)
    frame["is_term2"] = (
        pd.to_datetime(frame["date"]).dt.month < config.project.school_year_start_month
    ).astype(float)

    # Term-1 actuals: the final state of term-1 cumulative ratios, used as features in term 2
    term1_finals = (
        frame[frame["is_term2"] == 0]
        .sort_values("date")
        .groupby(["student_id", "academic_year"], as_index=False)
        .last()[["student_id", "academic_year", "term_cumulative_missed_ratio", "term_cumulative_giltig_ratio", "term_cumulative_ogiltig_ratio"]]
        .rename(columns={
            "term_cumulative_missed_ratio": "term1_actual_missed_ratio",
            "term_cumulative_giltig_ratio": "term1_actual_giltig_ratio",
            "term_cumulative_ogiltig_ratio": "term1_actual_ogiltig_ratio",
        })
    )
    frame = frame.merge(term1_finals, on=["student_id", "academic_year"], how="left")
    # In term-1 rows, term 1 is not yet complete so zero out the actuals
    for col in ("term1_actual_missed_ratio", "term1_actual_giltig_ratio", "term1_actual_ogiltig_ratio"):
        frame.loc[frame["is_term2"] == 0, col] = 0.0
        frame[col] = frame[col].fillna(0.0)

    # Year progress features from calendar
    calendar = prepared.calendar.copy()
    calendar["academic_year"] = academic_year_from_date(
        pd.to_datetime(calendar["date"]), config.project.school_year_start_month
    )
    year_calendar_totals = (
        calendar[calendar["is_instructional_day"]]
        .groupby(["school_id", "academic_year"], as_index=False)
        .agg(year_total_school_days=("date", "nunique"))
    )
    frame["year_school_days_elapsed"] = frame.groupby(["student_id", "academic_year"]).cumcount() + 1
    frame = frame.merge(year_calendar_totals, on=["school_id", "academic_year"], how="left")
    frame["year_progress_ratio"] = safe_divide(frame["year_school_days_elapsed"], frame["year_total_school_days"])
    frame["year_days_remaining"] = frame["year_total_school_days"] - frame["year_school_days_elapsed"]

    frame = _filter_years(frame, years)

    feature_columns = CHRONIC_LASYAR_NUMERIC_FEATURES + STUDENT_DAY_CATEGORICAL_FEATURES
    features = FeatureBundle(
        frame=frame,
        numeric_features=CHRONIC_LASYAR_NUMERIC_FEATURES,
        categorical_features=STUDENT_DAY_CATEGORICAL_FEATURES,
    )
    return TaskDataset(
        name=f"chronic_year_{int(threshold * 100)}",
        frame=_select_unique_columns(
            frame,
            ["date", "student_id", "school_id", "class_id", "grade", "stage", "term_id", "academic_year", target_column]
            + feature_columns,
        ),
        target_column=target_column,
        features=features,
        id_columns=["date", "student_id", "school_id", "class_id", "grade", "stage", "term_id", "academic_year"],
    )


def build_all_task_datasets(prepared: PreparedData, config: AppConfig, years: list[str] | None = None) -> dict[str, TaskDataset]:
    return {
        "short_horizon": build_short_horizon_dataset(prepared, config, years),
        "chronic_10": build_chronic_dataset(prepared, 0.10, years),
        "chronic_20": build_chronic_dataset(prepared, 0.20, years),
        "chronic_year_10": build_chronic_lasyar_dataset(prepared, 0.10, config, years),
        "chronic_year_20": build_chronic_lasyar_dataset(prepared, 0.20, config, years),
        "lesson": build_lesson_dataset(prepared, config, years),
    }
