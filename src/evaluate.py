"""Evaluation helpers focused on Macro-F1."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


def macro_f1(y_true: Iterable, y_pred: Iterable) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_f1(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "human_f1": float(f1[0]),
        "ai_f1": float(f1[1]),
        "human_support": int(support[0]),
        "ai_support": int(support[1]),
    }


def per_language_macro_f1(
    y_true: Iterable,
    y_pred: Iterable,
    languages: Iterable[str],
) -> dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    langs = np.asarray(list(languages))

    scores: dict[str, float] = {}
    for lang in sorted(set(langs)):
        mask = langs == lang
        if mask.sum() == 0:
            continue
        scores[lang] = macro_f1(y_true[mask], y_pred[mask])
    return scores


def best_threshold_for_macro_f1(
    y_true: Iterable,
    y_prob: Iterable,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    y_true = np.asarray(list(y_true))
    y_prob = np.asarray(list(y_prob))

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = macro_f1(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def print_report(
    y_true: Iterable,
    y_pred: Iterable,
    languages: Iterable[str] | None = None,
    title: str = "Evaluation",
) -> dict:
    metrics = {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **per_class_f1(y_true, y_pred),
    }
    if languages is not None:
        metrics["per_language_macro_f1"] = per_language_macro_f1(y_true, y_pred, languages)

    print(f"\n=== {title} ===")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Human F1:  {metrics['human_f1']:.4f}  (n={metrics['human_support']})")
    print(f"AI F1:     {metrics['ai_f1']:.4f}  (n={metrics['ai_support']})")

    if languages is not None:
        print("Per-language Macro-F1:")
        for lang, score in metrics["per_language_macro_f1"].items():
            print(f"  {lang}: {score:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["human", "ai"], digits=4))

    return metrics
