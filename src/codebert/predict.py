"""Generate predictions from a fine-tuned CodeBERT model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.codebert.inference import load_model, predict_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained CodeBERT model.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Parquet/CSV with at least 'id' and 'code' columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/codebert_submission.csv",
        help="Output CSV path (id, label).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/codebert.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config, default_name="codebert.yaml")
    model_cfg = config["model"]
    model_dir: Path = config["paths"]["model_dir"]
    model, tokenizer, threshold, _ = load_model(model_dir)

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

    results = predict_dataframe(
        df,
        model,
        tokenizer,
        threshold,
        max_length=model_cfg["max_length"],
        batch_size=model_cfg["eval_batch_size"],
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results[["id", "label"]].to_csv(output_path, index=False)
    print(f"Wrote {len(results):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
