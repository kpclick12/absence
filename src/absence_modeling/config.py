from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectConfig:
    output_dir: Path = Path("artifacts")
    random_seed: int = 42
    school_year_start_month: int = 8


@dataclass(slots=True)
class DataConfig:
    attendance_events: Path
    student_schedule: Path
    school_calendar: Path


@dataclass(slots=True)
class ModelingConfig:
    substantial_absence_threshold: float = 0.5
    short_horizon_school_days: int = 5
    lesson_horizon_school_days: int = 5
    top_k_fractions: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    minimum_rows_for_ebm: int = 500
    train_negative_to_positive_ratio: float = 5.0
    max_train_rows_per_task: int = 250_000


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    train_years: list[str]
    validation_years: list[str]
    test_years: list[str] = field(default_factory=list)
    production_refit_years: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    modeling: ModelingConfig
    experiments: list[ExperimentConfig]

    @property
    def output_dir(self) -> Path:
        return self.project.output_dir


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}

    project_raw = raw.get("project", {})
    data_raw = _require(raw, "data")
    modeling_raw = raw.get("modeling", {})
    experiments_raw = _require(raw, "experiments")

    project = ProjectConfig(
        output_dir=Path(project_raw.get("output_dir", "artifacts")),
        random_seed=int(project_raw.get("random_seed", 42)),
        school_year_start_month=int(project_raw.get("school_year_start_month", 8)),
    )

    data = DataConfig(
        attendance_events=Path(_require(data_raw, "attendance_events")),
        student_schedule=Path(_require(data_raw, "student_schedule")),
        school_calendar=Path(_require(data_raw, "school_calendar")),
    )

    modeling = ModelingConfig(
        substantial_absence_threshold=float(modeling_raw.get("substantial_absence_threshold", 0.5)),
        short_horizon_school_days=int(modeling_raw.get("short_horizon_school_days", 5)),
        lesson_horizon_school_days=int(modeling_raw.get("lesson_horizon_school_days", 5)),
        top_k_fractions=[float(value) for value in modeling_raw.get("top_k_fractions", [0.01, 0.02, 0.05])],
        minimum_rows_for_ebm=int(modeling_raw.get("minimum_rows_for_ebm", 500)),
        train_negative_to_positive_ratio=float(modeling_raw.get("train_negative_to_positive_ratio", 5.0)),
        max_train_rows_per_task=int(modeling_raw.get("max_train_rows_per_task", 250_000)),
    )

    experiments = [
        ExperimentConfig(
            name=item["name"],
            train_years=[str(value) for value in item["train_years"]],
            validation_years=[str(value) for value in item["validation_years"]],
            test_years=[str(value) for value in item.get("test_years", [])],
            production_refit_years=[str(value) for value in item.get("production_refit_years", [])],
        )
        for item in experiments_raw
    ]

    return AppConfig(project=project, data=data, modeling=modeling, experiments=experiments)
