"""Download HumanVsAICode JSONL files from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.config import load_config

DATASET_REPO = "OSS-forge/HumanVsAICode"
FILES = {
    "python": "python_dataset.jsonl",
    "java": "java_dataset.jsonl",
}


def download_language(lang: str, output_dir: Path) -> Path:
    if lang not in FILES:
        raise ValueError(f"Unknown language: {lang}. Choose from {list(FILES)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / FILES[lang]

    if target.exists():
        print(f"[skip] {target} already exists")
        return target

    print(f"Downloading {FILES[lang]} ...")
    cached = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=FILES[lang],
        repo_type="dataset",
    )
    shutil.copy2(cached, target)
    print(f"Saved to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HumanVsAICode dataset files.")
    parser.add_argument(
        "--lang",
        choices=["python", "java", "both"],
        default="both",
        help="Which language file(s) to download.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/baseline.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir: Path = config["paths"]["raw_dir"]

    langs = list(FILES) if args.lang == "both" else [args.lang]
    for lang in langs:
        download_language(lang, output_dir)


if __name__ == "__main__":
    main()
