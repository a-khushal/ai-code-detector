from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_dir: Path) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, float, dict]:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Missing {model_dir}. Run: python -m src.codebert.train"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    meta_path = model_dir / "meta.json"
    threshold = 0.5
    meta: dict = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        threshold = float(meta.get("threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, threshold, meta


def predict_proba(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    probs: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())

    return np.asarray(probs, dtype=float)


def predict_dataframe(
    df: pd.DataFrame,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    threshold: float,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    prob = predict_proba(model, tokenizer, df["code"].tolist(), max_length, batch_size)
    pred = (prob >= threshold).astype(int)
    return pd.DataFrame({"id": df["id"], "label": pred, "prob_ai": prob})
