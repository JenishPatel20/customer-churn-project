
# Customer Churn Prediction (Telco, SaaS, BankChurners) — XGBoost + SHAP

This repository contains an end-to-end churn prediction pipeline evaluated across **three datasets**:
1) **Telco Customer Churn** (telecommunications)  
2) **SaaS Media Churn** (engagement-based SaaS)  
3) **BankChurners** (credit-card attrition / banking)

The project provides:
- Training with reproducible splits
- Model evaluation (ROC curve + confusion matrix)
- Model explainability (SHAP bar + SHAP beeswarm)
- Saved artifacts (models, metrics, plots) for inclusion in the final paper

---

## Team 
- **Herik Patel**  
- **Jenish Patel**
- **Abhi Patel**

Instructor: **Prof. Dr. Ruixiang Tang**

---

## Repository Structure

```

customer-churn-project-main/
├─ data/
│  ├─ raw/
│  │  ├─ telco.csv
│  │  ├─ saas.csv
│  │  └─ bankchurners.csv
│  └─ processed/                 # (optional / if created by extensions)
├─ models/
│  ├─ telco_xgb.joblib
│  ├─ saas_xgb.joblib
│  └─ bankchurners_xgb.joblib
├─ reports/
│  ├─ metrics_telco_xgb.json
│  ├─ metrics_saas_xgb.json
│  ├─ metrics_bankchurners_xgb.json
│  ├─ split_telco.npz
│  ├─ split_saas.npz
│  ├─ split_bankchurners.npz
│  ├─ roc_telco_xgb.png
│  ├─ roc_saas_xgb.png
│  ├─ roc_bankchurners_xgb.png
│  ├─ cm_telco_xgb.png
│  ├─ cm_bankchurners_xgb.png
│  ├─ shap_beeswarm_telco_xgb.png
│  ├─ shap_beeswarm_saas_xgb.png
│  ├─ shap_beeswarm_bankchurners_xgb.png
│  └─ shap_bar_*.png             # (if generated)
└─ src/
├─ config.py
├─ data.py
├─ train.py
├─ eval.py
└─ explain.py

````

---

## Datasets

Place the CSV files under `data/raw/`:

- `data/raw/telco.csv`  
  - IBM Telco Customer Churn dataset (≈ 7k rows).
- `data/raw/saas.csv`  
  - SaaS engagement dataset (≈ 963 rows).
- `data/raw/bankchurners.csv`  
  - BankChurners / credit card attrition dataset (≈ 10k rows).

**Important:** The pipeline expects the dataset names:
- `telco`
- `saas`
- `bankchurners`

---

## Environment Setup

### 1) Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

### 2) Install dependencies

If you already have a `requirements.txt`, run:

```powershell
pip install -r requirements.txt
```

Otherwise, install minimum required packages:

```powershell
pip install -U pip
pip install pandas numpy scikit-learn xgboost shap matplotlib joblib category_encoders
```

Verify XGBoost:

```powershell
python -c "import xgboost; print(xgboost.__version__)"
```

---

## How to Run (Recommended Workflow)

For each dataset, run these three commands in order:

### A) Telco

```powershell
python -m src.train   --dataset telco --model xgb --tune
python -m src.eval    --dataset telco --model xgb
python -m src.explain --dataset telco --model xgb
```

### B) SaaS

```powershell
python -m src.train   --dataset saas --model xgb --tune
python -m src.eval    --dataset saas --model xgb
python -m src.explain --dataset saas --model xgb
```

### C) BankChurners

```powershell
python -m src.train   --dataset bankchurners --model xgb --tune
python -m src.eval    --dataset bankchurners --model xgb
python -m src.explain --dataset bankchurners --model xgb
```

---

## Outputs (What Gets Generated)

### Training (`src.train`)

* Saves model: `models/<dataset>_<model>.joblib`
* Saves metrics: `reports/metrics_<dataset>_<model>.json`
* Saves split indices: `reports/split_<dataset>.npz`

Metrics include:

* `cv_best_auc` (if tuning)
* `holdout_auc`
* `holdout_f1`
* `cv_best_params`

### Evaluation (`src.eval`)

* ROC curve: `reports/roc_<dataset>_<model>.png`
* Confusion matrix: `reports/cm_<dataset>_<model>.png`

### Explainability (`src.explain`)

* SHAP bar plot: `reports/shap_bar_<dataset>_<model>.png` (if enabled)
* SHAP beeswarm: `reports/shap_beeswarm_<dataset>_<model>.png`

---

## Notes on Results / AUC Target

* **Telco** and **BankChurners** typically achieve **AUC ≥ 0.85** with tuned XGBoost because they contain strong churn precursors (contract/tenure in Telco; inactivity/transaction behavior in banking).
* **SaaS** may produce lower AUC because the dataset is smaller and contains mostly engagement features, with limited direct pre-churn signals (e.g., no billing failures / inactivity duration). This is documented in the final paper’s **Limitations** section.

---

## Reproducibility

* Train/test split is stratified and uses a fixed seed.
* Cross-validation is stratified (5-fold).
* Saved `split_*.npz` ensures the exact evaluation split can be reused.

---

## Troubleshooting

### “invalid choice: 'bankchurners'”

Ensure `src/train.py`, `src/eval.py`, and `src/explain.py` all include:

```python
choices=["telco","saas","bankchurners"]
```

### Missing CSV error

Confirm the file exists and name matches exactly:

* `data/raw/bankchurners.csv`

### SHAP errors / slow runtime

SHAP can be compute-heavy. Run on CPU and ensure:

```powershell
pip install shap
```

---
