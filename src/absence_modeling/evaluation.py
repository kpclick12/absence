from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss


def precision_at_fraction(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, fraction: float) -> float:
    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(scores, dtype=float)
    if truth.size == 0:
        return np.nan
    cutoff = max(1, int(np.ceil(truth.size * fraction)))
    top_index = np.argsort(-predictions)[:cutoff]
    return float(np.mean(truth[top_index]))


def binary_metrics(
    frame: pd.DataFrame,
    target_column: str,
    score_column: str,
    fractions: list[float],
) -> dict[str, float]:
    y_true = frame[target_column].to_numpy(dtype=float)
    y_score = frame[score_column].to_numpy(dtype=float)
    metrics = {
        "row_count": float(len(frame)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    for fraction in fractions:
        metrics[f"precision_at_{fraction:.2f}"] = precision_at_fraction(y_true, y_score, fraction)
    return metrics


def compute_permutation_importance(
    model: Any,
    frame: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Compute permutation importance for a fitted model on the given frame.

    Returns a dict of feature_name -> mean importance score (decrease in precision@1%).
    Positive score = feature matters. Near-zero = can safely drop.
    """
    x = frame[feature_names].copy()
    y = frame[target_column].to_numpy(dtype=float)

    def scorer(estimator: Any, x_eval: pd.DataFrame, y_eval: np.ndarray) -> float:
        probs = estimator.predict_proba(x_eval)[:, 1]
        return precision_at_fraction(y_eval, probs, 0.05)

    result = sklearn_permutation_importance(
        model,
        x,
        y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return {name: float(score) for name, score in zip(feature_names, result.importances_mean, strict=True)}


def subgroup_metrics(
    frame: pd.DataFrame,
    group_column: str,
    target_column: str,
    score_column: str,
    fractions: list[float],
    min_rows: int = 25,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for group_value, subgroup in frame.groupby(group_column):
        if len(subgroup) < min_rows:
            continue
        metrics = binary_metrics(subgroup, target_column, score_column, fractions)
        metrics[group_column] = str(group_value)
        results.append(metrics)
    return results


@dataclass(slots=True)
class LessonAggregateResult:
    metrics: dict[str, float]
    aggregated_frame: pd.DataFrame


def lesson_aggregate_metrics(
    frame: pd.DataFrame,
    score_column: str,
    target_column: str,
    fractions: list[float],
) -> LessonAggregateResult:
    grouped = (
        frame.groupby(["date", "school_id", "class_id", "lesson_id"], as_index=False)
        .agg(
            expected_absence=(score_column, "sum"),
            actual_absence=(target_column, "sum"),
        )
        .sort_values(["date", "school_id", "class_id", "lesson_id"])
    )

    if grouped.empty:
        metrics = {"lesson_count": 0.0, "rank_correlation": np.nan}
        for fraction in fractions:
            metrics[f"lesson_hit_rate_at_{fraction:.2f}"] = np.nan
        return LessonAggregateResult(metrics=metrics, aggregated_frame=grouped)

    expected_rank = pd.Series(grouped["expected_absence"]).rank(method="average")
    actual_rank = pd.Series(grouped["actual_absence"]).rank(method="average")
    metrics = {
        "lesson_count": float(len(grouped)),
        "rank_correlation": float(expected_rank.corr(actual_rank)),
        "mae_expected_absence": float(np.mean(np.abs(grouped["expected_absence"] - grouped["actual_absence"]))),
    }
    for fraction in fractions:
        cutoff = max(1, int(np.ceil(len(grouped) * fraction)))
        predicted_top = set(grouped.nlargest(cutoff, "expected_absence").index.tolist())
        actual_top = set(grouped.nlargest(cutoff, "actual_absence").index.tolist())
        metrics[f"lesson_hit_rate_at_{fraction:.2f}"] = float(len(predicted_top.intersection(actual_top)) / cutoff)
    return LessonAggregateResult(metrics=metrics, aggregated_frame=grouped)
