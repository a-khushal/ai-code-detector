# ISEC AI Code Detector

Binary classifier (human=0, AI=1) for the [HumanVsAICode](https://huggingface.co/datasets/OSS-forge/HumanVsAICode) dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
src/common/      download, preprocess, evaluate (shared)
src/tfidf/       TF-IDF + Logistic Regression (CPU)
src/codebert/    reserved for later
configs/         data.yaml, tfidf.yaml
models/tfidf/    TF-IDF artifacts
models/codebert/ reserved for later
```

See `COMMANDS.md` for all run commands.

## Usage

```bash
python -m src.common.download_data --lang both
python -m src.common.preprocess --lang both --sample-fraction 0.01  # omit flag for full data
python -m src.tfidf.train
python -m src.tfidf.predict --input data/processed/test.parquet --output submissions/tfidf_submission.csv
```

Metric: Macro-F1.
