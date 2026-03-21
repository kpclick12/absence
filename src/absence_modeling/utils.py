from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator_array, np.nan, dtype=float)
    mask = denominator_array > 0
    result[mask] = numerator_array[mask] / denominator_array[mask]
    return result


def grade_to_stage(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "UNKNOWN"
    text = str(value).strip().upper()
    if text in {"F", "FK", "FORSKOLEKLASS", "FORSKOLEKLASSEN", "0"}:
        return "F-3"
    try:
        number = int(float(text))
    except ValueError:
        return "UNKNOWN"
    if number <= 3:
        return "F-3"
    if number <= 6:
        return "4-6"
    if number <= 9:
        return "7-9"
    return "UNKNOWN"


def academic_year_from_date(date_series: pd.Series, start_month: int = 8) -> pd.Series:
    years = date_series.dt.year.to_numpy()
    months = date_series.dt.month.to_numpy()
    start_year = np.where(months >= start_month, years, years - 1)
    end_year = (start_year + 1) % 100
    return pd.Series(
        [f"{int(start)}/{int(end):02d}" for start, end in zip(start_year, end_year, strict=False)],
        index=date_series.index,
    )


def previous_academic_year(label: str) -> str | None:
    try:
        start_text, end_text = label.split("/")
        start_year = int(start_text)
        end_year = int(end_text)
    except ValueError:
        return None
    return f"{start_year - 1}/{(end_year - 1) % 100:02d}"


def start_minutes(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    timestamp = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(timestamp):
        return np.nan
    return float(timestamp.hour * 60 + timestamp.minute)


def time_bin_from_minutes(minutes: float) -> str:
    if np.isnan(minutes):
        return "unknown"
    if minutes < 9 * 60:
        return "before_09"
    if minutes < 11 * 60:
        return "09_11"
    if minutes < 13 * 60:
        return "11_13"
    return "after_13"


@dataclass(slots=True)
class FeatureBundle:
    frame: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    if output_path.suffix == ".csv":
        frame.to_csv(output_path, index=False)
        return
    frame.to_parquet(output_path, index=False)
