# Commands

Activate venv first:

```bash
cd /home/a-khushal/Desktop/ISEC
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (shared)

```bash
# Download raw JSONL (~1.1 GB)
python -m src.common.download_data --lang both

# Preprocess — full dataset
python -m src.common.preprocess --lang both

# Preprocess — quick dev run (1%)
python -m src.common.preprocess --lang both --sample-fraction 0.01

# Preprocess — one language only
python -m src.common.preprocess --lang python
python -m src.common.preprocess --lang java
```

## TF-IDF (CPU)

```bash
# Train
python -m src.tfidf.train

# Predict / submission
python -m src.tfidf.predict \
  --input data/processed/test.parquet \
  --output submissions/tfidf_submission.csv
```

## CodeBERT (GPU)

```bash
pip install -r requirements-codebert.txt

# Train (defaults to 50k train samples — edit configs/codebert.yaml for full data)
python -m src.codebert.train

# Predict / submission
python -m src.codebert.predict \
  --input data/processed/test.parquet \
  --output submissions/codebert_submission.csv
```

## Kaggle notebook

```python
# Setup (Internet ON)
!git clone https://github.com/a-khushal/ai-code-detector.git
%cd ai-code-detector
!pip install -r requirements.txt

# TF-IDF pipeline
!python -m src.common.download_data --lang both
!python -m src.common.preprocess --lang both
!python -m src.tfidf.train
!python -m src.tfidf.predict --input data/processed/test.parquet --output submissions/tfidf_submission.csv

# CodeBERT (Session options → GPU ON)
!pip install -r requirements-codebert.txt
!python -m src.codebert.train
!python -m src.codebert.predict --input data/processed/test.parquet --output submissions/codebert_submission.csv
```

## Score submission locally

```python
import pandas as pd
from sklearn.metrics import f1_score, classification_report

test = pd.read_parquet("data/processed/test.parquet")
preds = pd.read_csv("submissions/tfidf_submission.csv")
merged = test.merge(preds, on="id", suffixes=("_true", "_pred"))

print("Macro-F1:", f1_score(merged["label_true"], merged["label_pred"], average="macro"))
print(classification_report(merged["label_true"], merged["label_pred"], target_names=["human", "ai"]))
```

## Config files

| File | Used by |
|------|---------|
| `configs/data.yaml` | download, preprocess |
| `configs/tfidf.yaml` | TF-IDF train/predict |
| `configs/codebert.yaml` | CodeBERT train/predict |
