"""Fine-tune CodeBERT for human vs AI code classification."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.common.config import load_config
from src.common.data import load_split
from src.common.evaluate import best_threshold_for_macro_f1, print_report
from src.codebert.inference import predict_proba


def maybe_subsample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def build_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_dict({"code": df["code"].tolist(), "label": df["label"].tolist()})

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["code"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=["code"])


def save_meta(model_dir: Path, threshold: float, metrics: dict, config: dict) -> None:
    meta = {
        "threshold": threshold,
        "metrics": metrics,
        "model": config["model"],
    }
    with (model_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/codebert.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config, default_name="codebert.yaml")
    model_cfg = config["model"]
    processed_dir: Path = config["paths"]["processed_dir"]
    model_dir: Path = config["paths"]["model_dir"]
    seed = config["seed"]

    train_df = maybe_subsample(
        load_split(processed_dir, "train"),
        max_samples=model_cfg.get("max_train_samples"),
        seed=seed,
    )
    val_df = load_split(processed_dir, "val")
    test_df = load_split(processed_dir, "test")

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Test samples:  {len(test_df):,}")

    model_name = model_cfg["name"]
    max_length = model_cfg["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = build_dataset(train_df, tokenizer, max_length)
    val_dataset = build_dataset(val_df, tokenizer, max_length)

    use_fp16 = bool(model_cfg.get("fp16", True) and torch.cuda.is_available())
    if model_cfg.get("fp16", True) and not torch.cuda.is_available():
        print("CUDA not available — training on CPU (slow). Enable GPU on Kaggle.")

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        learning_rate=model_cfg["learning_rate"],
        per_device_train_batch_size=model_cfg["batch_size"],
        per_device_eval_batch_size=model_cfg["eval_batch_size"],
        num_train_epochs=model_cfg["num_train_epochs"],
        weight_decay=model_cfg["weight_decay"],
        warmup_ratio=model_cfg["warmup_ratio"],
        gradient_accumulation_steps=model_cfg.get("gradient_accumulation_steps", 1),
        eval_strategy="steps",
        eval_steps=model_cfg["eval_steps"],
        logging_steps=model_cfg["logging_steps"],
        save_strategy="steps",
        save_steps=model_cfg["eval_steps"],
        save_total_limit=model_cfg.get("save_total_limit", 1),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        seed=seed,
        report_to=[],
    )

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": float(f1_score(labels, preds, average="macro"))}

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print(f"Training CodeBERT ({model_name}) ...")
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    val_prob = predict_proba(
        model,
        tokenizer,
        val_df["code"].tolist(),
        max_length=max_length,
        batch_size=model_cfg["eval_batch_size"],
    )
    threshold, val_macro_f1 = best_threshold_for_macro_f1(val_df["label"], val_prob)
    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = print_report(
        val_df["label"],
        val_pred,
        languages=val_df["lang"],
        title=f"Validation (threshold={threshold:.2f})",
    )

    test_prob = predict_proba(
        model,
        tokenizer,
        test_df["code"].tolist(),
        max_length=max_length,
        batch_size=model_cfg["eval_batch_size"],
    )
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = print_report(
        test_df["label"],
        test_pred,
        languages=test_df["lang"],
        title=f"Test (threshold={threshold:.2f})",
    )

    save_meta(
        model_dir=model_dir,
        threshold=threshold,
        metrics={
            "val": val_metrics,
            "test": test_metrics,
            "val_macro_f1_at_threshold": val_macro_f1,
        },
        config=config,
    )
    print(f"\nSaved model to {model_dir}")


if __name__ == "__main__":
    main()
