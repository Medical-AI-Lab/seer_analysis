# Transformer-Based Foundation Models for Mortality Prediction Across 142 Rare Cancers

This repository contains the analysis code used for our study on 5-year mortality prediction across 142 rare cancers using TabPFN (a tabular foundation model) compared with logistic regression, random forest, and XGBoost.

> **Status**: Pre-review (manuscript under submission). Code currently runs on *simulated data* to reproduce the full pipeline end-to-end without distributing SEER data.

## Repository structure
```
.
├── analysis.py
├── requirements.txt
└── README.md
```

## What’s in `analysis.py`
- Builds a synthetic (dummy) dataset that mimics the study design (multi-cancer, geographic dev/test split).
- Encodes categorical features with `OrdinalEncoder`.
- Chooses an **adaptive CV strategy** based on minimum class size (StratifiedKFold or StratifiedShuffleSplit).
- Trains and evaluates **TabPFN**, **LogisticRegression**, **RandomForest**, **XGBoost** and writes `analysis_results.csv`.

> The current script runs on CPU by default and sets PyTorch to single-thread for reproducibility and stability in CI.

## Quick start

### 1) Environment
- Python ≥ 3.10 recommended

```bash
# create & activate a clean environment (example with venv)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### 2) Run the pipeline (with dummy data)
```bash
python analysis.py
```
Outputs:
- Console logs with per-cancer/model AUC (OOF dev, test ensemble)
- `analysis_results.csv` saved to the repo root

## Using real SEER data (notes)
This repository does **not** include SEER data. To reproduce the study on real data:
1. Obtain SEER Research Plus data under the standard Data-Use Agreement.
2. Export the tab-delimited table from SEER*Stat with the variables and years described in the paper.
3. Replace the dummy-data creation step with your SEER table loader, keeping the same column schema / encoders.
4. Preserve the **geographic hold-out** split and the **pre-treatment** variable set to avoid post-treatment bias.

> Tip: Keep categorical value mappings stable across dev/test; unseen categories are handled via `OrdinalEncoder(..., unknown_value=-1)` in this pipeline.

## Reproducibility tips
- Fix random seeds (already set to 42 across models where applicable).
- Run on a single thread for PyTorch (already enforced).
- Avoid heavy hyperparameter searches for small-n cohorts; defaults are used intentionally.
