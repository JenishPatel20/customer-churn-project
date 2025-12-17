# src/train.py
from __future__ import annotations

import argparse
import json
## from dataclasses import asdict
from typing import Dict, Any
import numpy as np

import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from .config import MODELS_DIR, REPORTS_DIR
from .data import load_dataset


def build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "logreg":
        return LogisticRegression(max_iter=3000)
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    if model_name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        )
    raise ValueError("model must be one of: logreg, rf, xgb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="telco", choices=["telco", "saas"])
    ap.add_argument("--model", default="xgb", choices=["logreg", "rf", "xgb"])
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    bundle = load_dataset(args.dataset)

    idx = bundle.X.index.to_numpy()

    idx_train, idx_test = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=42,
        stratify=bundle.y,
    )

    X_train = bundle.X.loc[idx_train]
    y_train = bundle.y[bundle.X.index.get_indexer(idx_train)]
    X_test = bundle.X.loc[idx_test]
    y_test = bundle.y[bundle.X.index.get_indexer(idx_test)]

    clf = build_model(args.model)

    pipe = Pipeline(
        steps=[
            ("pre", bundle.preprocessor),
            ("clf", clf),
        ]
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "auc": float(roc_auc_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{args.dataset}_{args.model}.joblib"
    joblib.dump(pipe, model_path)

    metrics_path = REPORTS_DIR / f"metrics_{args.dataset}_{args.model}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    split_path = REPORTS_DIR / f"split_{args.dataset}.npz"
    np.savez(split_path, train_idx=idx_train, test_idx=idx_test)

    print("Saved split:", split_path)
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
