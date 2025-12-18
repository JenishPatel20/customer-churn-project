# src/train.py
from __future__ import annotations

import argparse
import json
from typing import Dict, Any

import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
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


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)


def build_model(model_name: str, y_train: np.ndarray | None = None):
    model_name = model_name.lower()

    if model_name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced")

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=900,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if model_name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Run: pip install xgboost")

        spw = _compute_scale_pos_weight(y_train) if y_train is not None else 1.0

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            tree_method="hist",
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            reg_alpha=0.0,
            gamma=0.0,
            scale_pos_weight=spw,
        )

    raise ValueError("model must be one of: logreg, rf, xgb")


def tune_xgb(pipe: Pipeline, X_train, y_train, dataset_name: str) -> tuple[Pipeline, Dict[str, Any]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    d = dataset_name.strip().lower()

    if d == "saas":
        # SaaS is smaller/noisier: allow more flexibility, but keep it bounded
        param_dist = {
            "clf__n_estimators": [800, 1200, 1600, 2200, 3200],
            "clf__max_depth": [2, 3, 4, 5, 6],
            "clf__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "clf__min_child_weight": [1, 2, 3, 5, 8],
            "clf__reg_lambda": [0.5, 1.0, 2.0, 4.0],
            "clf__reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "clf__gamma": [0.0, 0.5, 1.0, 2.0],
        }
        n_iter = 120

    elif d in ("bankchurners", "bank"):
        # BankChurners typically has strong signal; moderate search is enough
        param_dist = {
            "clf__n_estimators": [600, 900, 1200, 1600, 2200],
            "clf__max_depth": [2, 3, 4, 5],
            "clf__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "clf__min_child_weight": [1, 2, 3, 5],
            "clf__reg_lambda": [0.5, 1.0, 2.0, 4.0],
            "clf__gamma": [0.0, 0.5, 1.0],
        }
        n_iter = 80

    else:
        # Telco: stable and close to the proposal threshold; smaller grid for speed
        param_dist = {
            "clf__n_estimators": [800, 1200, 1600, 2200],
            "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.01, 0.02, 0.03],
            "clf__subsample": [0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "clf__min_child_weight": [1, 2, 3, 5],
            "clf__reg_lambda": [0.5, 1.0, 2.0],
            "clf__gamma": [0.0, 0.5, 1.0],
        }
        n_iter = 80

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    info = {
        "cv_best_auc": float(search.best_score_),
        "cv_best_params": search.best_params_,
    }
    return search.best_estimator_, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="telco", choices=["telco", "saas", "bankchurners"])
    ap.add_argument("--model", default="xgb", choices=["logreg", "rf", "xgb"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--tune", action="store_true")
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

    clf = build_model(args.model, y_train=y_train)

    pipe = Pipeline(
        steps=[
            ("pre", bundle.preprocessor),
            ("clf", clf),
        ]
    )

    tuning_info: Dict[str, Any] = {}

    if args.model == "xgb" and args.tune:
        pipe, tuning_info = tune_xgb(pipe, X_train, y_train, dataset_name=args.dataset)
    else:
        pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "holdout_auc": float(roc_auc_score(y_test, probs)),
        "holdout_f1": float(f1_score(y_test, preds)),
        **tuning_info,
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
