from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig
from .contracts import ATTENDANCE_COLUMNS, CALENDAR_COLUMNS, SCHEDULE_COLUMNS, validate_columns


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format for {path}")


def load_inputs(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attendance = _load_table(config.attendance_events)
    schedule = _load_table(config.student_schedule)
    calendar = _load_table(config.school_calendar)

    validate_columns("attendance_events", attendance.columns, ATTENDANCE_COLUMNS)
    validate_columns("student_schedule", schedule.columns, SCHEDULE_COLUMNS)
    validate_columns("school_calendar", calendar.columns, CALENDAR_COLUMNS)

    return attendance, schedule, calendar
