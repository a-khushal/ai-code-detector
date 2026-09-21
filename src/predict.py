"""Generate binary predictions / submission CSV from a trained baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config


def load_model(model_dir: Path) -> tuple[object, float]:
    pipeline_path = model_dir / "baseline_pipeline.joblib"
    meta_path = model_dir / "baseline_meta.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Missing {pipeline_path}. Run: python -m src.train"
        )

    pipeline = joblib.load(pipeline_path)
    threshold = 0.5
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        threshold = float(meta.get("threshold", 0.5))

    return pipeline, threshold


def predict_dataframe(df: pd.DataFrame, pipeline, threshold: float) -> pd.DataFrame:
    prob = pipeline.predict_proba(df["code"])[:, 1]
    pred = (prob >= threshold).astype(int)
    return pd.DataFrame(
        {
            "id": df["id"],
            "label": pred,
            "prob_ai": prob,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained baseline.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Parquet/CSV with at least 'id' and 'code' columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/submission.csv",
        help="Output CSV path (id, label).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/baseline.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir: Path = config["paths"]["model_dir"]
    pipeline, threshold = load_model(model_dir)

    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if "code" not in df.columns:
        raise ValueError("Input must contain a 'code' column.")
    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df["id"] = df.index.astype(str)

    results = predict_dataframe(df, pipeline, threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results[["id", "label"]].to_csv(output_path, index=False)
    print(f"Wrote {len(results):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
