"""Flatten paired HumanVsAICode rows into binary-labeled samples with group splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config

AI_COLUMNS = {
    "chatgpt": "chatgpt_code",
    "deepseek": "dsc_code",
    "qwen": "qwen_code",
}


def is_valid_code(code: object, min_chars: int) -> bool:
    if not isinstance(code, str):
        return False
    stripped = code.strip()
    return len(stripped) >= min_chars


def flatten_rows(df: pd.DataFrame, lang: str, min_chars: int) -> pd.DataFrame:
    records: list[dict] = []

    for row in df.itertuples(index=False):
        hm_index = row.hm_index

        if is_valid_code(row.human_code, min_chars):
            records.append(
                {
                    "id": hm_index,
                    "hm_index": hm_index,
                    "code": row.human_code.strip(),
                    "label": 0,
                    "lang": lang,
                    "source_model": "human",
                }
            )

        for model_name, col in AI_COLUMNS.items():
            code = getattr(row, col)
            if is_valid_code(code, min_chars):
                records.append(
                    {
                        "id": f"{hm_index}_{model_name}",
                        "hm_index": hm_index,
                        "code": code.strip(),
                        "label": 1,
                        "lang": lang,
                        "source_model": model_name,
                    }
                )

    return pd.DataFrame.from_records(records)


def load_paired_jsonl(path: Path, sample_fraction: float | None = None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if sample_fraction is not None and 0 < sample_fraction < 1:
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
    return df


def split_by_group(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    groups = df[["hm_index", "lang"]].drop_duplicates()
    train_groups, temp_groups = train_test_split(
        groups,
        test_size=(val_frac + test_frac),
        random_state=seed,
        stratify=groups["lang"],
    )
    relative_test = test_frac / (val_frac + test_frac)
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_groups["lang"],
    )

    train = df[df["hm_index"].isin(train_groups["hm_index"])].copy()
    val = df[df["hm_index"].isin(val_groups["hm_index"])].copy()
    test = df[df["hm_index"].isin(test_groups["hm_index"])].copy()
    return train, val, test


def process_language(
    raw_path: Path,
    lang: str,
    processed_dir: Path,
    min_chars: int,
    splits: dict[str, float],
    seed: int,
    sample_fraction: float | None = None,
) -> dict[str, Path]:
    print(f"Processing {lang} from {raw_path} ...")
    paired = load_paired_jsonl(raw_path, sample_fraction=sample_fraction)
    flat = flatten_rows(paired, lang=lang, min_chars=min_chars)

    train, val, test = split_by_group(
        flat,
        train_frac=splits["train"],
        val_frac=splits["val"],
        test_frac=splits["test"],
        seed=seed,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": processed_dir / f"{lang}_train.parquet",
        "val": processed_dir / f"{lang}_val.parquet",
        "test": processed_dir / f"{lang}_test.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test.to_parquet(paths["test"], index=False)

    print(
        f"  {lang}: {len(paired):,} paired rows -> "
        f"{len(flat):,} flat samples "
        f"(train={len(train):,}, val={len(val):,}, test={len(test):,})"
    )
    return paths


def combine_splits(processed_dir: Path, split: str) -> Path:
    frames = []
    for lang in ("python", "java"):
        path = processed_dir / f"{lang}_{split}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(f"No processed files found for split '{split}' in {processed_dir}")

    combined = pd.concat(frames, ignore_index=True)
    out = processed_dir / f"{split}.parquet"
    combined.to_parquet(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess HumanVsAICode into binary splits.")
    parser.add_argument(
        "--lang",
        choices=["python", "java", "both"],
        default="both",
        help="Which language file(s) to preprocess.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional fraction of paired rows to use (e.g. 0.01 for quick dev runs).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/baseline.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir: Path = config["paths"]["raw_dir"]
    processed_dir: Path = config["paths"]["processed_dir"]
    splits = config["splits"]
    seed = config["seed"]
    min_chars = config["min_code_chars"]

    langs = ["python", "java"] if args.lang == "both" else [args.lang]
    for lang in langs:
        raw_path = raw_dir / f"{lang}_dataset.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing {raw_path}. Run: python -m src.download_data --lang {lang}"
            )
        process_language(
            raw_path=raw_path,
            lang=lang,
            processed_dir=processed_dir,
            min_chars=min_chars,
            splits=splits,
            seed=seed,
            sample_fraction=args.sample_fraction,
        )

    if args.lang == "both":
        for split in ("train", "val", "test"):
            out = combine_splits(processed_dir, split)
            print(f"Combined {split} -> {out}")


if __name__ == "__main__":
    main()
