from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import AppConfig
from .evaluation import binary_metrics
from .tasks import TaskDataset

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:  # pragma: no cover - optional dependency
    ExplainableBoostingClassifier = None


@dataclass(slots=True)
class ProbabilityCalibrator:
    constant: float | None = None
    model: LogisticRegression | None = None

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> None:
        if probabilities.size == 0:
            self.constant = 0.0
            return
        unique = np.unique(target)
        if unique.size < 2:
            self.constant = float(unique[0]) if unique.size == 1 else 0.0
            return
        self.model = LogisticRegression(solver="lbfgs", max_iter=200)
        self.model.fit(probabilities.reshape(-1, 1), target)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        if self.constant is not None:
            return np.full(probabilities.shape[0], self.constant, dtype=float)
        return probabilities


@dataclass(slots=True)
class TrainedModel:
    task_name: str
    candidate_name: str
    model: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]
    stage_calibrators: dict[str, ProbabilityCalibrator]
    global_calibrator: ProbabilityCalibrator
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.model.predict_proba(frame[self.numeric_features + self.categorical_features])[:, 1]
        stages = frame["stage"].astype(str).to_numpy()
        calibrated = np.empty_like(probabilities, dtype=float)
        for stage in np.unique(stages):
            mask = stages == stage
            calibrator = self.stage_calibrators.get(stage, self.global_calibrator)
            calibrated[mask] = calibrator.predict(probabilities[mask])
        return calibrated

    def dump(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TrainedModel":
        return joblib.load(path)


def _logistic_pipeline(task: TaskDataset, random_seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                task.features.numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                task.features.categorical_features,
            ),
        ]
    )
    classifier = LogisticRegression(
        solver="saga",
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=random_seed,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def _ebm_pipeline(task: TaskDataset, random_seed: int) -> Pipeline | None:
    if ExplainableBoostingClassifier is None:
        return None
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), task.features.numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                task.features.categorical_features,
            ),
        ],
        sparse_threshold=0.0,
    )
    classifier = ExplainableBoostingClassifier(interactions=0, random_state=random_seed)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def _candidate_pipeline(candidate_name: str, task: TaskDataset, random_seed: int) -> Pipeline:
    if candidate_name == "logistic":
        return _logistic_pipeline(task, random_seed)
    if candidate_name == "ebm":
        pipeline = _ebm_pipeline(task, random_seed)
        if pipeline is None:
            raise ValueError("EBM candidate requested but the optional dependency is not installed.")
        return pipeline
    raise ValueError(f"Unknown candidate model: {candidate_name}")


def _sample_training_rows(frame: pd.DataFrame, target_column: str, config: AppConfig) -> pd.DataFrame:
    positive = frame[frame[target_column] == 1]
    negative = frame[frame[target_column] == 0]
    if positive.empty or negative.empty:
        sampled = frame.copy()
    else:
        max_negative = int(min(len(negative), len(positive) * config.modeling.train_negative_to_positive_ratio))
        negative_sample = negative.sample(
            n=max_negative,
            random_state=config.project.random_seed,
            replace=False,
        )
        sampled = pd.concat([positive, negative_sample], ignore_index=True)
    if len(sampled) > config.modeling.max_train_rows_per_task:
        positive = sampled[sampled[target_column] == 1]
        negative = sampled[sampled[target_column] == 0]
        remaining = max(config.modeling.max_train_rows_per_task - len(positive), 0)
        if remaining and not negative.empty:
            negative = negative.sample(
                n=min(len(negative), remaining),
                random_state=config.project.random_seed,
                replace=False,
            )
        sampled = pd.concat([positive, negative], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=config.project.random_seed).reset_index(drop=True)


def _fit_calibrators(predictions: np.ndarray, validation_frame: pd.DataFrame, target_column: str) -> tuple[dict[str, ProbabilityCalibrator], ProbabilityCalibrator]:
    stage_calibrators: dict[str, ProbabilityCalibrator] = {}
    global_calibrator = ProbabilityCalibrator()
    global_calibrator.fit(predictions, validation_frame[target_column].to_numpy())
    for stage, subgroup in validation_frame.groupby("stage"):
        calibrator = ProbabilityCalibrator()
        mask = validation_frame["stage"].astype(str).eq(str(stage)).to_numpy()
        calibrator.fit(predictions[mask], subgroup[target_column].to_numpy())
        stage_calibrators[str(stage)] = calibrator
    return stage_calibrators, global_calibrator


def train_task_model(
    task: TaskDataset,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: AppConfig,
) -> tuple[TrainedModel, list[dict[str, Any]]]:
    x_columns = task.features.numeric_features + task.features.categorical_features
    train_sample = _sample_training_rows(train_frame, task.target_column, config)

    candidates: list[tuple[str, Pipeline]] = [("logistic", _logistic_pipeline(task, config.project.random_seed))]
    ebm_pipeline = _ebm_pipeline(task, config.project.random_seed)
    if ebm_pipeline is not None and len(train_sample) >= config.modeling.minimum_rows_for_ebm:
        candidates.append(("ebm", ebm_pipeline))

    candidate_records: list[dict[str, Any]] = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_score = -np.inf

    for candidate_name, pipeline in candidates:
        pipeline.fit(train_sample[x_columns], train_sample[task.target_column])
        validation_probs = pipeline.predict_proba(validation_frame[x_columns])[:, 1]
        metrics = binary_metrics(validation_frame.assign(raw_score=validation_probs), task.target_column, "raw_score", config.modeling.top_k_fractions)
        candidate_records.append({"candidate": candidate_name, **metrics})
        selection_metric = metrics.get(f"precision_at_{config.modeling.top_k_fractions[0]:.2f}", np.nan)
        if np.isnan(selection_metric):
            selection_metric = -np.inf
        if selection_metric > best_score:
            best_score = float(selection_metric)
            best_name = candidate_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError(f"No candidate model could be trained for task {task.name}")

    validation_probs = best_pipeline.predict_proba(validation_frame[x_columns])[:, 1]
    stage_calibrators, global_calibrator = _fit_calibrators(validation_probs, validation_frame, task.target_column)
    trained = TrainedModel(
        task_name=task.name,
        candidate_name=best_name,
        model=best_pipeline,
        numeric_features=task.features.numeric_features,
        categorical_features=task.features.categorical_features,
        stage_calibrators=stage_calibrators,
        global_calibrator=global_calibrator,
        metadata={"candidate_records": candidate_records},
    )
    return trained, candidate_records


def refit_selected_model(
    task: TaskDataset,
    candidate_name: str,
    refit_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    config: AppConfig,
) -> TrainedModel:
    x_columns = task.features.numeric_features + task.features.categorical_features
    fit_sample = _sample_training_rows(refit_frame, task.target_column, config)
    pipeline = _candidate_pipeline(candidate_name, task, config.project.random_seed)
    pipeline.fit(fit_sample[x_columns], fit_sample[task.target_column])
    calibration_probs = pipeline.predict_proba(calibration_frame[x_columns])[:, 1]
    stage_calibrators, global_calibrator = _fit_calibrators(calibration_probs, calibration_frame, task.target_column)
    return TrainedModel(
        task_name=task.name,
        candidate_name=candidate_name,
        model=pipeline,
        numeric_features=task.features.numeric_features,
        categorical_features=task.features.categorical_features,
        stage_calibrators=stage_calibrators,
        global_calibrator=global_calibrator,
    )
