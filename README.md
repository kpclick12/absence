# Absence Modeling

This repository contains a reproducible Python pipeline for building three municipality-wide student absence models:

- short-horizon student absence risk over the next `5` school days by default
- chronic risk by term end at `10%` and `20%` missed scheduled minutes
- lesson-level risk for the coming school week

The pipeline is built around the data contracts agreed in the planning phase:

- `attendance_events`
- `student_schedule`
- `school_calendar`

It trains global models, calibrates probabilities separately by school stage (`F-3`, `4-6`, `7-9`), evaluates them on time-based year splits, and writes model artifacts plus prediction tables that can be filtered to a single class.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

To enable the optional Explainable Boosting Machine challenger:

```bash
uv pip install -e ".[dev,ebm]"
```

## Configuration

Use [`configs/example.yaml`](/Users/johanhellenas/Desktop/projects_codex/absence/configs/example.yaml) as the starting point. Point it at your actual `csv` or `parquet` files.

Model-specific documentation:

- [Modeling Principles](/Users/johanhellenas/Desktop/projects_codex/absence/docs/modeling_principles.md)
- [Short-Horizon Student Risk Model](/Users/johanhellenas/Desktop/projects_codex/absence/docs/short_horizon_model.md)
- [Chronic Risk Model](/Users/johanhellenas/Desktop/projects_codex/absence/docs/chronic_model.md)

## Train and Evaluate

```bash
absence-model run --config configs/example.yaml
```

Outputs are written under `artifacts/` by default:

- `artifacts/reports/metrics.json`
- `artifacts/reports/metrics.md`
- `artifacts/models/`
- `artifacts/predictions/`

## Score a Class

```bash
absence-model score-class \
  --config configs/example.yaml \
  --as-of-date 2026-02-12 \
  --school-id SCHOOL_1 \
  --class-id CLASS_7A
```

This writes:

- short-horizon probabilities for students in the selected class
- chronic `10%` and `20%` probabilities for the class
- student-level lesson probabilities for the coming week
- aggregated expected absence per upcoming lesson

## Notes

- The pipeline excludes lessons with missing attendance registration from modeling labels and denominators.
- Final probabilities are calibrated by stage after model selection.
- Future pre-registered absences are not used as predictors.
- If you only have one prior year, the pipeline still runs, but the evaluation is weaker and should be treated as feasibility evidence rather than production evidence.
