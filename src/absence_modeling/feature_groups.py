"""
Feature group definitions for the experiment search loop.

Instead of selecting individual features (which leads to 40-50 feature configs),
candidates specify which named groups to include. Each group covers a coherent
set of related features (3-10 features). Claude edits the group list between runs.

Usage:
    numeric, categorical = resolve_features(["rolling_short", "streaks", "prior_year"], "chronic_10")
"""
from __future__ import annotations

# These numeric features are always included regardless of group selection.
# They provide fundamental temporal and grade context that every model needs.
ALWAYS_INCLUDED_NUMERIC: list[str] = [
    "grade_numeric",
    "school_day_index",
    "term_progress_ratio",
    "term_days_remaining",
]

# Categorical features are always included (small, stable, no iteration needed).
ALWAYS_INCLUDED_CATEGORICAL: list[str] = [
    "stage",
    "weekday_name",
]

# Named feature groups. Each group is a coherent set of related signals.
# Estimated feature count per group shown in comments.
FEATURE_GROUPS: dict[str, list[str]] = {
    # Daily raw attendance signals (8 features)
    "daily_raw": [
        "daily_recorded_lessons",
        "daily_recorded_minutes",
        "daily_missed_minutes",
        "daily_missed_ratio",
        "daily_giltig_ratio",
        "daily_ogiltig_ratio",
        "daily_partial_lesson_ratio",
        "daily_substantial_miss_lessons",
    ],
    # Short rolling windows 1-5 days (15 features: 5 metrics × 3 windows)
    "rolling_short": [
        "roll_1d_missed_ratio",
        "roll_3d_missed_ratio",
        "roll_5d_missed_ratio",
        "roll_1d_giltig_ratio",
        "roll_3d_giltig_ratio",
        "roll_5d_giltig_ratio",
        "roll_1d_ogiltig_ratio",
        "roll_3d_ogiltig_ratio",
        "roll_5d_ogiltig_ratio",
        "roll_1d_partial_ratio",
        "roll_3d_partial_ratio",
        "roll_5d_partial_ratio",
        "roll_1d_substantial_day_rate",
        "roll_3d_substantial_day_rate",
        "roll_5d_substantial_day_rate",
    ],
    # Long rolling windows 10-20 days (10 features: 5 metrics × 2 windows)
    "rolling_long": [
        "roll_10d_missed_ratio",
        "roll_20d_missed_ratio",
        "roll_10d_giltig_ratio",
        "roll_20d_giltig_ratio",
        "roll_10d_ogiltig_ratio",
        "roll_20d_ogiltig_ratio",
        "roll_10d_partial_ratio",
        "roll_20d_partial_ratio",
        "roll_10d_substantial_day_rate",
        "roll_20d_substantial_day_rate",
    ],
    # Streak and recency signals (4 features)
    "streaks": [
        "substantial_streak_days",
        "ogiltig_streak_days",
        "school_days_since_substantial_miss",
        "school_days_since_ogiltig_miss",
    ],
    # Short vs long-window trend (1 feature)
    "trend": [
        "trend_5_vs_20_missed_ratio",
    ],
    # Term-to-date cumulative ratios (3 features)
    "term_cumulative": [
        "term_cumulative_missed_ratio",
        "term_cumulative_giltig_ratio",
        "term_cumulative_ogiltig_ratio",
    ],
    # Prior year annual summary (5 features)
    "prior_year": [
        "has_prior_year_history",
        "prior_year_missed_ratio",
        "prior_year_giltig_ratio",
        "prior_year_ogiltig_ratio",
        "prior_year_substantial_day_rate",
    ],
    # Calendar gap features - breaks, weekends (3 features)
    "calendar": [
        "calendar_days_since_previous_school_day",
        "calendar_days_until_next_school_day",
        "after_long_break",
    ],
    # Distance to chronic thresholds - chronic tasks only (2 features)
    "gap_to_threshold": [
        "gap_to_10pct",
        "gap_to_20pct",
    ],
    # Upcoming week schedule load - short_horizon only (7 features)
    "schedule_future": [
        "future_school_days_available",
        "future_total_scheduled_lessons",
        "future_total_scheduled_minutes",
        "future_average_daily_minutes",
        "future_total_subject_count",
        "future_morning_lessons",
        "future_afternoon_lessons",
    ],
    # Next school day details - short_horizon only (9 features)
    "schedule_next_day": [
        "next_school_day_total_scheduled_lessons",
        "next_school_day_total_scheduled_minutes",
        "next_school_day_first_lesson_start",
        "next_school_day_last_lesson_start",
        "next_school_day_subject_count",
        "next_school_day_morning_lessons",
        "next_school_day_afternoon_lessons",
        "next_school_day_gap_days",
        "next_school_day_after_long_break",
    ],
    # School-year level features - chronic_year tasks only (11 features)
    "year_level": [
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
    ],
    # Target lesson details - lesson task only (9 features)
    "lesson_target": [
        "horizon_school_days",
        "calendar_days_until_target",
        "target_scheduled_minutes",
        "target_lesson_start_minutes",
        "target_day_total_scheduled_lessons",
        "target_day_total_scheduled_minutes",
        "target_day_subject_count",
        "target_days_since_previous_school_day",
        "target_after_long_break",
    ],
    # Historical subject/time/weekday patterns - lesson task only (8 features)
    "lesson_history": [
        "hist_subject_lesson_count",
        "hist_subject_substantial_rate",
        "hist_weekday_lesson_count",
        "hist_weekday_substantial_rate",
        "hist_timebin_lesson_count",
        "hist_timebin_substantial_rate",
        "hist_subject_weekday_timebin_lesson_count",
        "hist_subject_weekday_timebin_substantial_rate",
    ],
}

# Default starting groups per task (all groups valid for that task).
# Use these as the baseline "candidate 001" config for each task.
TASK_DEFAULT_GROUPS: dict[str, list[str]] = {
    "short_horizon": [
        "daily_raw", "rolling_short", "rolling_long", "streaks", "trend",
        "term_cumulative", "prior_year", "calendar",
        "schedule_future", "schedule_next_day",
    ],
    "chronic_10": [
        "daily_raw", "rolling_short", "rolling_long", "streaks", "trend",
        "term_cumulative", "prior_year", "calendar", "gap_to_threshold",
    ],
    "chronic_20": [
        "daily_raw", "rolling_short", "rolling_long", "streaks", "trend",
        "term_cumulative", "prior_year", "calendar", "gap_to_threshold",
    ],
    "chronic_year_10": [
        "daily_raw", "rolling_short", "rolling_long", "streaks", "trend",
        "term_cumulative", "prior_year", "calendar", "gap_to_threshold", "year_level",
    ],
    "chronic_year_20": [
        "daily_raw", "rolling_short", "rolling_long", "streaks", "trend",
        "term_cumulative", "prior_year", "calendar", "gap_to_threshold", "year_level",
    ],
    "lesson": [
        "daily_raw", "rolling_short", "rolling_long", "streaks", "trend",
        "term_cumulative", "prior_year", "calendar",
        "lesson_target", "lesson_history",
    ],
}

# Extra categorical features required by specific tasks (on top of ALWAYS_INCLUDED_CATEGORICAL)
TASK_EXTRA_CATEGORICALS: dict[str, list[str]] = {
    "short_horizon": ["next_school_day_weekday_name"],
    "lesson": ["target_subject", "target_weekday_name", "target_lesson_time_bin",
               "target_first_lesson_flag", "target_last_lesson_flag"],
}


def resolve_features(groups: list[str], task_name: str) -> tuple[list[str], list[str]]:
    """
    Resolve a list of group names into (numeric_features, categorical_features).

    Always prepends ALWAYS_INCLUDED_NUMERIC and ALWAYS_INCLUDED_CATEGORICAL.
    Unknown group names raise ValueError.

    Returns deduplicated lists preserving order.
    """
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}. Valid: {sorted(FEATURE_GROUPS)}")

    seen: set[str] = set()
    numeric: list[str] = []
    for feature in ALWAYS_INCLUDED_NUMERIC:
        if feature not in seen:
            seen.add(feature)
            numeric.append(feature)
    for group in groups:
        for feature in FEATURE_GROUPS[group]:
            if feature not in seen:
                seen.add(feature)
                numeric.append(feature)

    categorical = list(ALWAYS_INCLUDED_CATEGORICAL)
    for extra in TASK_EXTRA_CATEGORICALS.get(task_name, []):
        if extra not in categorical:
            categorical.append(extra)

    return numeric, categorical


def group_importances(feature_importance: dict[str, float]) -> dict[str, float]:
    """
    Aggregate per-feature importance scores into per-group scores (mean).
    Features not belonging to any named group are collected under "base".
    """
    group_features: dict[str, list[float]] = {name: [] for name in FEATURE_GROUPS}
    group_features["base"] = []

    named = {f: g for g, feats in FEATURE_GROUPS.items() for f in feats}
    for feature, score in feature_importance.items():
        group = named.get(feature, "base")
        group_features[group].append(score)

    return {
        g: float(sum(scores) / len(scores)) if scores else 0.0
        for g, scores in group_features.items()
        if scores
    }
