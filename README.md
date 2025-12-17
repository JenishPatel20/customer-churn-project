# Customer Churn Prediction (Course Final Project)

This project predicts customer churn using supervised learning models (baseline + boosted trees) and includes evaluation + interpretability.

## Repo Structure
- `src/` : main code (data prep, training, evaluation, explainability)
- `notebooks/` : EDA / experiments
- `data/raw/` : raw dataset files (NOT committed)
- `data/processed/` : cleaned/processed datasets (NOT committed)
- `models/` : saved models (ignored)
- `reports/` : figures/tables for the paper

## Setup

### 1) Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

