# src/eval.py
from __future__ import annotations

import argparse
import joblib
import matplotlib.pyplot as plt


from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from .config import MODELS_DIR, REPORTS_DIR
from .data import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="telco", choices=["telco","saas","bankchurners"])
    ap.add_argument("--model", default="xgb", choices=["logreg", "rf", "xgb"])
    ## ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    bundle = load_dataset(args.dataset)

    import numpy as np


    split_path = REPORTS_DIR / f"split_{args.dataset}.npz"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Run training first:\n"
            f"python -m src.train --dataset {args.dataset} --model {args.model}"
        )

    split = np.load(split_path, allow_pickle=True)
    idx_test = split["test_idx"]

    X_test = bundle.X.loc[idx_test]
    y_test = bundle.y[bundle.X.index.get_indexer(idx_test)]

    model_path = MODELS_DIR / f"{args.dataset}_{args.model}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}. Train first:\n"
            f"python -m src.train --dataset {args.dataset} --model {args.model}"
        )


    pipe = joblib.load(model_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure()
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    roc_out = REPORTS_DIR / f"roc_{args.dataset}_{args.model}.png"
    plt.title(f"ROC - {args.dataset} ({args.model})")
    plt.savefig(roc_out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", roc_out)

    # Confusion Matrix
    plt.figure()
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
    cm_out = REPORTS_DIR / f"cm_{args.dataset}_{args.model}.png"
    plt.title(f"Confusion Matrix - {args.dataset} ({args.model})")
    plt.savefig(cm_out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", cm_out)


if __name__ == "__main__":
    main()
