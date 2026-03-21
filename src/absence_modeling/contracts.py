from __future__ import annotations

from collections.abc import Iterable


ATTENDANCE_COLUMNS = {
    "student_id",
    "school_id",
    "class_id",
    "grade",
    "term_id",
    "date",
    "lesson_id",
    "subject",
    "scheduled_start_at",
    "scheduled_minutes",
    "attended_minutes",
    "missed_minutes",
    "absence_validity",
    "record_created_at",
    "record_updated_at",
}

SCHEDULE_COLUMNS = {
    "student_id",
    "school_id",
    "class_id",
    "grade",
    "date",
    "lesson_id",
    "subject",
    "scheduled_start_at",
    "scheduled_minutes",
}

CALENDAR_COLUMNS = {
    "school_id",
    "date",
    "term_id",
    "is_instructional_day",
    "school_day_index",
}


def validate_columns(name: str, actual: Iterable[str], required: set[str]) -> None:
    missing = sorted(required.difference(set(actual)))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{name} is missing required columns: {missing_str}")
