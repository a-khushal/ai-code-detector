# ISEC AI Code Detector

Binary classifier (human=0, AI=1) for the [HumanVsAICode](https://huggingface.co/datasets/OSS-forge/HumanVsAICode) dataset. Baseline: TF-IDF + Logistic Regression (CPU only).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.download_data --lang both
python -m src.preprocess --lang both --sample-fraction 0.01  # omit flag for full data
python -m src.train
python -m src.predict --input data/processed/test.parquet --output submissions/submission.csv
```

Metric: Macro-F1. Config: `configs/baseline.yaml`.
