from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "baseline.yaml"
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.setdefault("paths", {})
    for key in ("raw_dir", "processed_dir", "model_dir"):
        if key in paths:
            paths[key] = (PROJECT_ROOT / paths[key]).resolve()

    return config
