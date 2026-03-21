from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


def _school_days(start: str, count: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start=start, periods=count))


def _attendance_pattern(student_id: str, day_index: int, subject: str) -> tuple[int, str]:
    if student_id == "S1" and subject == "Math" and day_index % 5 == 0:
        return 45, "ogiltig"
    if student_id == "S1" and subject == "Swedish" and day_index % 5 == 4:
        return 23, "giltig"
    if student_id == "S3" and day_index >= 5:
        return (45 if subject == "Math" else 23), "ogiltig"
    if student_id == "S4" and day_index % 6 == 2 and subject == "Math":
        return 30, "ogiltig"
    return 0, "present"


def _build_synthetic_data(base_dir: Path) -> dict[str, Path]:
    years = {
        "2022/23": _school_days("2022-08-22", 10),
        "2023/24": _school_days("2023-08-21", 10),
        "2024/25": _school_days("2024-08-19", 10),
        "2025/26": _school_days("2025-08-18", 10),
    }
    students = [
        ("S1", "CLASS_7A", "7"),
        ("S2", "CLASS_7A", "7"),
        ("S3", "CLASS_7A", "7"),
        ("S4", "CLASS_4A", "4"),
        ("S5", "CLASS_4A", "4"),
    ]
    lessons = [("L1", "Math", "08:00"), ("L2", "Swedish", "10:00")]

    schedule_rows: list[dict] = []
    attendance_rows: list[dict] = []
    calendar_rows: list[dict] = []

    for academic_year, dates in years.items():
        term_id = f"{academic_year}_term"
        for school_day_index, date in enumerate(dates, start=1):
            calendar_rows.append(
                {
                    "school_id": "SCHOOL_1",
                    "date": date,
                    "term_id": term_id,
                    "is_instructional_day": True,
                    "school_day_index": school_day_index,
                }
            )
            for student_id, class_id, grade in students:
                for lesson_suffix, subject, start_time in lessons:
                    lesson_id = f"{academic_year}_{student_id}_{date.strftime('%Y%m%d')}_{lesson_suffix}"
                    schedule_rows.append(
                        {
                            "student_id": student_id,
                            "school_id": "SCHOOL_1",
                            "class_id": class_id,
                            "grade": grade,
                            "date": date,
                            "lesson_id": lesson_id,
                            "subject": subject,
                            "scheduled_start_at": f"{date.date()} {start_time}",
                            "scheduled_minutes": 45,
                        }
                    )

                    if academic_year == "2025/26" and date > pd.Timestamp("2025-08-27"):
                        continue
                    if student_id == "S2" and academic_year == "2024/25" and date == dates[2] and subject == "Swedish":
                        continue

                    missed_minutes, absence_validity = _attendance_pattern(student_id, school_day_index - 1, subject)
                    attendance_rows.append(
                        {
                            "student_id": student_id,
                            "school_id": "SCHOOL_1",
                            "class_id": class_id,
                            "grade": grade,
                            "term_id": term_id,
                            "date": date,
                            "lesson_id": lesson_id,
                            "subject": subject,
                            "scheduled_start_at": f"{date.date()} {start_time}",
                            "scheduled_minutes": 45,
                            "attended_minutes": 45 - missed_minutes,
                            "missed_minutes": missed_minutes,
                            "absence_validity": absence_validity,
                            "record_created_at": f"{date.date()} 15:00",
                            "record_updated_at": f"{date.date()} 15:05",
                        }
                    )

    attendance = pd.DataFrame(attendance_rows)
    schedule = pd.DataFrame(schedule_rows)
    calendar = pd.DataFrame(calendar_rows)

    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    attendance_path = data_dir / "attendance_events.parquet"
    schedule_path = data_dir / "student_schedule.parquet"
    calendar_path = data_dir / "school_calendar.parquet"
    attendance.to_parquet(attendance_path, index=False)
    schedule.to_parquet(schedule_path, index=False)
    calendar.to_parquet(calendar_path, index=False)
    return {
        "attendance": attendance_path,
        "schedule": schedule_path,
        "calendar": calendar_path,
    }


def _write_config(base_dir: Path, data_paths: dict[str, Path]) -> Path:
    config = {
        "project": {
            "output_dir": str(base_dir / "artifacts"),
            "random_seed": 7,
            "school_year_start_month": 8,
        },
        "data": {
            "attendance_events": str(data_paths["attendance"]),
            "student_schedule": str(data_paths["schedule"]),
            "school_calendar": str(data_paths["calendar"]),
        },
        "modeling": {
            "substantial_absence_threshold": 0.5,
            "lesson_horizon_school_days": 5,
            "top_k_fractions": [0.1, 0.2],
            "minimum_rows_for_ebm": 5000,
            "train_negative_to_positive_ratio": 4.0,
            "max_train_rows_per_task": 10000,
        },
        "experiments": [
            {"name": "backtest_a", "train_years": ["2022/23"], "validation_years": ["2023/24"]},
            {"name": "backtest_b", "train_years": ["2022/23", "2023/24"], "validation_years": ["2024/25"]},
            {
                "name": "final_test",
                "train_years": ["2022/23", "2023/24"],
                "validation_years": ["2024/25"],
                "test_years": ["2025/26"],
                "production_refit_years": ["2022/23", "2023/24", "2024/25"],
            },
        ],
    }
    config_path = base_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> dict[str, Path]:
    data_paths = _build_synthetic_data(tmp_path)
    config_path = _write_config(tmp_path, data_paths)
    return {"config": config_path, **data_paths, "root": tmp_path}
