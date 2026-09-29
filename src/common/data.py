from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocess first, e.g. "
            f"python -m src.common.preprocess --lang both"
        )
    return pd.read_parquet(path)
