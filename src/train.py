"""Train a CPU-friendly TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.evaluate import best_threshold_for_macro_f1, print_report


def load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocess first, e.g. "
            f"python -m src.preprocess --lang both"
        )
    return pd.read_parquet(path)


def build_pipeline(config: dict) -> Pipeline:
    tfidf_cfg = config["tfidf"]
    model_cfg = config["model"]

    vectorizer = TfidfVectorizer(
        analyzer=tfidf_cfg["analyzer"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        max_features=tfidf_cfg["max_features"],
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        class_weight=model_cfg.get("class_weight"),
        max_iter=model_cfg.get("max_iter", 1000),
        C=model_cfg.get("C", 1.0),
        n_jobs=-1,
        random_state=config["seed"],
    )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )


def save_artifacts(
    pipeline: Pipeline,
    threshold: float,
    metrics: dict,
    model_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "baseline_pipeline.joblib")

    meta = {
        "threshold": threshold,
        "metrics": metrics,
    }
    with (model_dir / "baseline_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/baseline.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir: Path = config["paths"]["processed_dir"]
    model_dir: Path = config["paths"]["model_dir"]

    train_df = load_split(processed_dir, "train")
    val_df = load_split(processed_dir, "val")
    test_df = load_split(processed_dir, "test")

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Test samples:  {len(test_df):,}")

    pipeline = build_pipeline(config)
    print("Training baseline pipeline (CPU-only) ...")
    pipeline.fit(train_df["code"], train_df["label"])

    val_prob = pipeline.predict_proba(val_df["code"])[:, 1]
    threshold, val_macro_f1 = best_threshold_for_macro_f1(val_df["label"], val_prob)
    val_pred = (val_prob >= threshold).astype(int)

    val_metrics = print_report(
        val_df["label"],
        val_pred,
        languages=val_df["lang"],
        title=f"Validation (threshold={threshold:.2f})",
    )

    test_prob = pipeline.predict_proba(test_df["code"])[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = print_report(
        test_df["label"],
        test_pred,
        languages=test_df["lang"],
        title=f"Test (threshold={threshold:.2f})",
    )

    save_artifacts(
        pipeline=pipeline,
        threshold=threshold,
        metrics={
            "val": val_metrics,
            "test": test_metrics,
            "val_macro_f1_at_threshold": val_macro_f1,
        },
        model_dir=model_dir,
    )
    print(f"\nSaved model to {model_dir}")


if __name__ == "__main__":
    main()
