# src/explain.py
from __future__ import annotations

import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt

from .config import MODELS_DIR, REPORTS_DIR
from .data import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="telco", choices=["telco","saas","bankchurners"])
    ap.add_argument("--model", default="xgb", choices=["logreg", "rf", "xgb"])
    ap.add_argument("--max_rows", type=int, default=2000)
    args = ap.parse_args()

    import shap  # keep import here so basic training works even if shap isn't installed

    bundle = load_dataset(args.dataset)

    model_path = MODELS_DIR / f"{args.dataset}_{args.model}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}. Train first:\n"
            f"python -m src.train --dataset {args.dataset} --model {args.model}"
        )

    pipe = joblib.load(model_path)

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Explain the SAME test split used in eval (reproducible + no leakage)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = REPORTS_DIR / f"split_{args.dataset}.npz"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Run training first:\n"
            f"python -m src.train --dataset {args.dataset} --model {args.model}"
        )


    split = np.load(split_path, allow_pickle=True)
    idx_test = split["test_idx"]

    X = bundle.X.loc[idx_test]
    if len(X) > args.max_rows:
        X = X.sample(args.max_rows, random_state=42)

    Xt = pre.transform(X)

    # Feature names (best-effort)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = None

    # SHAP explainer depending on model type
    if args.model == "xgb":
        explainer = shap.TreeExplainer(clf)
    elif args.model == "rf":
        explainer = shap.TreeExplainer(clf)
    else:
        # logistic regression
        explainer = shap.LinearExplainer(clf, Xt)

    # Some transformers produce sparse matrices; SHAP is happier with dense for small datasets
    try:
        Xt_dense = Xt.toarray()
    except Exception:
        Xt_dense = Xt

    shap_values = explainer.shap_values(Xt_dense)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import re

    def pretty_feature_name(s: str) -> str:
        # Example inputs:
        #   "cat__Contract_Month-to-month"
        #   "num__MonthlyCharges"
        #   "cat__PaymentMethod_Bank transfer (automatic)"
        if s is None:
            return ""

        # strip transformer prefix
        s = re.sub(r"^(cat__|num__)", "", s)

        # turn one-hot into "Feature = Value"
        if "_" in s:
            parts = s.split("_", 1)
            if len(parts) == 2:
                feat, val = parts
                # common cleanup
                feat = feat.replace("SeniorCitizen", "Senior Citizen")
                feat = feat.replace("MonthlyCharges", "Monthly Charges")
                feat = feat.replace("TotalCharges", "Total Charges")
                feat = feat.replace("PaperlessBilling", "Paperless Billing")
                feat = feat.replace("MultipleLines", "Multiple Lines")
                feat = feat.replace("InternetService", "Internet Service")
                feat = feat.replace("OnlineSecurity", "Online Security")
                feat = feat.replace("OnlineBackup", "Online Backup")
                feat = feat.replace("TechSupport", "Tech Support")
                feat = feat.replace("StreamingTV", "Streaming TV")
                feat = feat.replace("StreamingMovies", "Streaming Movies")

                # make the value nicer
                val = val.replace("Yes", "Yes").replace("No", "No")
                return f"{feat} = {val}"

        # numeric columns (no one-hot)
        s = s.replace("SeniorCitizen", "Senior Citizen")
        s = s.replace("MonthlyCharges", "Monthly Charges")
        s = s.replace("TotalCharges", "Total Charges")
        s = s.replace("PaperlessBilling", "Paperless Billing")
        return s

    # Build pretty names
    pretty_names = None
    if feat_names is not None:
        pretty_names = [pretty_feature_name(str(x)) for x in feat_names]

    # If SHAP returns a list (some classifiers do), pick the positive class
    sv = shap_values
    if isinstance(sv, list) and len(sv) > 1:
        sv = sv[1]

    # Bar summary plot: larger figure + bigger fonts + tight layout
    fig = plt.figure(figsize=(12, 7))

    shap.summary_plot(
        sv,
        Xt_dense,
        feature_names=pretty_names if pretty_names is not None else feat_names,
        show=False,
        plot_type="bar",
        max_display=15,
    )

    ax = plt.gca()

    # Better title + axis labels
    fig.suptitle("Top Features Driving Churn (SHAP)", fontsize=16, y=0.97)

    ax.set_xlabel("Average impact on predicted churn risk (mean |SHAP value|)", fontsize=12)
    ax.xaxis.set_label_coords(0.1, -0.08)
    ax.set_ylabel("")  # remove default y-label (it’s not helpful)

    # Center and give space for long labels
    fig.subplots_adjust(left=0.40, right=0.95, top=0.88, bottom=0.12)

    out = REPORTS_DIR / f"shap_bar_{args.dataset}_{args.model}.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # Also save the beeswarm summary plot (more informative than bar)
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(12, 8))

    shap.summary_plot(
        sv,
        Xt_dense,
        feature_names=pretty_names if pretty_names is not None else feat_names,
        show=False,
        max_display=15,
    )

    ax = plt.gca()

    # Clearer axis meaning
    ax.set_xlabel("SHAP value (→ increases predicted churn risk, ← decreases predicted churn risk)", fontsize=12)

    # Title + short context
    ax.set_title("Model explanation (SHAP summary)", fontsize=16, pad=12)

    # Add a small caption to explain dots/colors
    fig.text(
        0.02,
        0.02,
        "How to read: Each dot = one customer. Position shows how much that feature pushes the prediction.\n"
        "Color shows feature value (blue = low, red = high). For rows like 'Feature = Value', red usually means that value is present.",
        fontsize=10,
    )

    # Optional: add a small legend (in addition to the colorbar)
    low_patch = mpatches.Patch(color="#1f77b4", label="Low feature value")
    high_patch = mpatches.Patch(color="#d62728", label="High feature value")
    ax.legend(handles=[low_patch, high_patch], loc="lower right", frameon=True, fontsize=10)

    # Give more room for long y-axis labels and the title/caption
    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))

    out2 = REPORTS_DIR / f"shap_beeswarm_{args.dataset}_{args.model}.png"
    plt.savefig(out2, dpi=250, bbox_inches="tight")
    plt.close()
    print("Saved:", out2)


if __name__ == "__main__":
    main()
