# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from .config import TELCO_CSV, SAAS_CSV


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    preprocessor: ColumnTransformer


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols: List[str] = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols: List[str] = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )


def load_telco(path: Path = TELCO_CSV) -> DatasetBundle:
    if not path.exists():
        raise FileNotFoundError(f"Missing Telco CSV at: {path}")

    df = pd.read_csv(path)

    # Fix common Telco issue: TotalCharges sometimes stored as string / spaces
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" not in df.columns:
        raise ValueError("Telco dataset must contain 'Churn' column.")

    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()
    drop_cols = ["Churn"]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    X = df.drop(columns=drop_cols)

    pre = _build_preprocessor(X)
    return DatasetBundle(name="telco", X=X, y=y, preprocessor=pre)


def load_saas_csv(path: Path = SAAS_CSV) -> DatasetBundle:
    """
    This is a generic SaaS loader. Put your SaaS dataset at data/raw/saas.csv.

    It will try to find a churn-like target column automatically.
    If it can't, it will raise an error and tell you what columns it sees.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing SaaS CSV at: {path}")

    df = pd.read_csv(path)

    # Try common churn label names
    candidates = ["Churn", "churn", "is_churn", "IsChurn", "cancelled", "canceled", "Cancelled", "Canceled"]
    target = next((c for c in candidates if c in df.columns), None)
    if target is None:
        raise ValueError(
            "Could not find churn label column in SaaS CSV. "
            f"Expected one of {candidates}. Columns seen: {list(df.columns)}"
        )

    # Normalize label to 0/1
    col = df[target]
    if col.dtype == "bool":
        y = col.astype(int).to_numpy()
    elif np.issubdtype(col.dtype, np.number):
        y = (col.fillna(0).astype(float) > 0).astype(int).to_numpy()
    else:
        y = col.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "churn", "cancelled", "canceled"]).astype(int).to_numpy()

    X = df.drop(columns=[target])
    pre = _build_preprocessor(X)
    return DatasetBundle(name="saas", X=X, y=y, preprocessor=pre)


def load_dataset(name: str) -> DatasetBundle:
    name = name.strip().lower()
    if name == "telco":
        return load_telco()
    if name == "saas":
        return load_saas_csv()
    raise ValueError("dataset must be one of: telco, saas")
