"""Task-specific evaluation metrics for FD and RUL."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(value)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(logits)
    return values / np.sum(values, axis=1, keepdims=True).clip(min=1e-12)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + end - 1) / 2.0 + 1.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum = original_ranks[positive].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def classification_metrics(
    labels: object,
    scores_or_logits: object,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Return Accuracy, macro F1, macro AUROC (OVR), and macro Recall.

    Missing classes in a validation split are ignored for macro averages while
    a split containing a single class reports AUROC as NaN, which is more
    honest than manufacturing a perfect ROC curve.
    """

    truth = _to_numpy(labels).reshape(-1).astype(np.int64)
    scores = _to_numpy(scores_or_logits)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if scores.shape[0] != truth.shape[0]:
        raise ValueError("labels and scores have different numbers of samples")
    if scores.shape[1] == 1:
        probabilities = np.concatenate((1.0 - 1.0 / (1.0 + np.exp(-scores)), 1.0 / (1.0 + np.exp(-scores))), axis=1)
    else:
        row_sums = scores.sum(axis=1)
        probabilities = scores if np.all(scores >= 0) and np.allclose(row_sums, 1.0, atol=1e-4) else _softmax(scores)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    accuracy = float(np.mean(predictions == truth)) if truth.size else float("nan")
    class_count = int(num_classes or probabilities.shape[1])
    active = sorted(set(truth.tolist()) | set(predictions.tolist()))
    f1_values = []
    recall_values = []
    auc_values = []
    for class_id in active:
        true_positive = int(np.sum((truth == class_id) & (predictions == class_id)))
        false_positive = int(np.sum((truth != class_id) & (predictions == class_id)))
        false_negative = int(np.sum((truth == class_id) & (predictions != class_id)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
        recall_values.append(recall)
        if class_id < probabilities.shape[1]:
            auc_values.append(_binary_auc((truth == class_id).astype(np.int64), probabilities[:, class_id]))
    finite_auc = [value for value in auc_values if np.isfinite(value)]
    return {
        "accuracy": accuracy,
        "f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "auroc": float(np.mean(finite_auc)) if finite_auc else float("nan"),
        "recall": float(np.mean(recall_values)) if recall_values else float("nan"),
        "num_samples": float(truth.size),
        "num_classes": float(class_count),
    }


def rul_metrics(targets: object, predictions: object) -> Dict[str, float]:
    """Return MAE, MSE, and RMSE on the target's declared scale."""

    truth = _to_numpy(targets).reshape(-1).astype(np.float64)
    pred = _to_numpy(predictions).reshape(-1).astype(np.float64)
    if truth.shape != pred.shape:
        raise ValueError("targets and predictions have different shapes")
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not np.any(valid):
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "num_samples": 0.0}
    error = pred[valid] - truth[valid]
    mse = float(np.mean(error * error))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "num_samples": float(valid.sum()),
    }


def grouped_classification_metrics(
    labels: Sequence[object], scores: Sequence[object], groups: Sequence[str]
) -> Dict[str, float]:
    """Compute per-dataset FD metrics and a macro aggregate."""

    if not (len(labels) == len(scores) == len(groups)):
        raise ValueError("labels, scores, and groups must have equal lengths")
    grouped: Dict[str, list] = defaultdict(list)
    for truth, score, group in zip(labels, scores, groups):
        grouped[str(group)].append((truth, score))
    per_group: Dict[str, Dict[str, float]] = {}
    for group, values in grouped.items():
        per_group[group] = classification_metrics(
            np.concatenate([_to_numpy(value[0]).reshape(-1) for value in values]),
            np.concatenate([_to_numpy(value[1]).reshape(1, -1) for value in values], axis=0),
        )
    result: Dict[str, float] = {}
    for metric in ("accuracy", "f1", "auroc", "recall"):
        values = [item[metric] for item in per_group.values() if np.isfinite(item[metric])]
        result[metric] = float(np.mean(values)) if values else float("nan")
    return result

