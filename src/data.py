# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import category_encoders as ce

from .config import TELCO_CSV, SAAS_CSV, BANKCHURNERS_CSV


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    preprocessor: Any  # can be ColumnTransformer or a custom transformer


ID_COLS: Set[str] = {
    "customerid", "customer_id", "customerid", "customerID", "CustomerID",
    "id", "ID", "Id",
    "clientnum", "CLIENTNUM",
}


def _drop_id_cols(X: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = []
    for c in X.columns:
        if c in ID_COLS or c.lower() in ID_COLS:
            cols_to_drop.append(c)
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop, errors="ignore")
    return X


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _add_engineered_features(dataset_name: str, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    d = dataset_name.strip().lower()

    if d == "telco":
        if "MonthlyCharges" in X.columns and "tenure" in X.columns:
            X["monthly_per_tenure"] = _safe_div(X["MonthlyCharges"], X["tenure"])
        if "TotalCharges" in X.columns and "tenure" in X.columns:
            X["total_per_tenure"] = _safe_div(X["TotalCharges"], X["tenure"])

    if d == "saas":
        if "TotalCharges" in X.columns and "AccountAge" in X.columns:
            X["charges_per_account_age"] = _safe_div(X["TotalCharges"], X["AccountAge"])
        if "MonthlyCharges" in X.columns and "AccountAge" in X.columns:
            X["monthly_per_account_age"] = _safe_div(X["MonthlyCharges"], X["AccountAge"])
        if "ViewingHoursPerWeek" in X.columns and "AccountAge" in X.columns:
            X["viewing_per_account_age"] = _safe_div(X["ViewingHoursPerWeek"], X["AccountAge"])
        if "ViewingHoursPerWeek" in X.columns and "MonthlyCharges" in X.columns:
            X["viewing_per_monthly_charge"] = _safe_div(X["ViewingHoursPerWeek"], X["MonthlyCharges"])
        if "SupportTicketsPerMonth" in X.columns and "ViewingHoursPerWeek" in X.columns:
            X["tickets_per_viewing_hour"] = _safe_div(X["SupportTicketsPerMonth"], X["ViewingHoursPerWeek"])
        if "ContentDownloadsPerMonth" in X.columns and "ViewingHoursPerWeek" in X.columns:
            X["downloads_per_viewing_hour"] = _safe_div(X["ContentDownloadsPerMonth"], X["ViewingHoursPerWeek"])

    if d in ("bankchurners", "bank"):
        # Useful ratios that often help tree models
        if "Total_Trans_Amt" in X.columns and "Total_Trans_Ct" in X.columns:
            X["avg_trans_amt"] = _safe_div(X["Total_Trans_Amt"], X["Total_Trans_Ct"])
        if "Total_Ct_Chng_Q4_Q1" in X.columns and "Total_Amt_Chng_Q4_Q1" in X.columns:
            X["ct_amt_chng_ratio"] = _safe_div(X["Total_Ct_Chng_Q4_Q1"], X["Total_Amt_Chng_Q4_Q1"])

    return X


def _build_preprocessor_ohe(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols: List[str] = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols: List[str] = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


class TargetEncodePreprocessor(BaseEstimator, TransformerMixin):
    """
    Robust target encoding that works with sklearn Pipeline/RandomizedSearch.
    Avoids ColumnTransformer column-name issues by handling cat+num in one transformer.
    """
    def __init__(self):
        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.num_imputer_ = SimpleImputer(strategy="median", add_indicator=True)
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent", add_indicator=True)
        self.te_ = ce.TargetEncoder(handle_unknown="value", handle_missing="value")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        # Fit numeric imputer
        _ = self.num_imputer_.fit(X[self.num_cols_])

        # Fit cat imputer + target encoder
        X_cat_imp = self.cat_imputer_.fit_transform(X[self.cat_cols_])
        # Build column names including missing-indicator columns produced by SimpleImputer(add_indicator=True)
        cat_feature_names = list(self.cat_cols_)
        if hasattr(self.cat_imputer_, "indicator_") and self.cat_imputer_.indicator_ is not None:
            missing_idx = self.cat_imputer_.indicator_.features_
            cat_feature_names += [f"{self.cat_cols_[i]}_missing" for i in missing_idx]

        X_cat_df = pd.DataFrame(X_cat_imp, columns=cat_feature_names)
        self.te_.fit(X_cat_df, y)
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_num = self.num_imputer_.transform(X[self.num_cols_])

        X_cat_imp = self.cat_imputer_.transform(X[self.cat_cols_])
        cat_feature_names = list(self.cat_cols_)
        if hasattr(self.cat_imputer_, "indicator_") and self.cat_imputer_.indicator_ is not None:
            missing_idx = self.cat_imputer_.indicator_.features_
            cat_feature_names += [f"{self.cat_cols_[i]}_missing" for i in missing_idx]

        X_cat_df = pd.DataFrame(X_cat_imp, columns=cat_feature_names)
        X_cat_te = self.te_.transform(X_cat_df).to_numpy()

        return np.hstack([X_num, X_cat_te])


def load_telco(path: Path = TELCO_CSV) -> DatasetBundle:
    if not path.exists():
        raise FileNotFoundError(f"Missing Telco CSV at: {path}")

    df = pd.read_csv(path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" not in df.columns:
        raise ValueError("Telco dataset must contain 'Churn' column.")

    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()

    X = df.drop(columns=["Churn"], errors="ignore")
    X = _drop_id_cols(X)
    X = _add_engineered_features("telco", X)

    pre = _build_preprocessor_ohe(X)
    return DatasetBundle(name="telco", X=X, y=y, preprocessor=pre)


def load_saas_csv(path: Path = SAAS_CSV) -> DatasetBundle:
    if not path.exists():
        raise FileNotFoundError(f"Missing SaaS CSV at: {path}")

    df = pd.read_csv(path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" not in df.columns:
        raise ValueError("SaaS dataset must contain 'Churn' column (0/1).")

    y = (df["Churn"].fillna(0).astype(int) > 0).astype(int).to_numpy()

    X = df.drop(columns=["Churn"], errors="ignore")
    X = _drop_id_cols(X)
    X = _add_engineered_features("saas", X)

    # Target encoding preprocessor (works reliably; avoids your previous TargetEncoder error)
    pre = TargetEncodePreprocessor()
    return DatasetBundle(name="saas", X=X, y=y, preprocessor=pre)


def load_bankchurners(path: Path = BANKCHURNERS_CSV) -> DatasetBundle:
    if not path.exists():
        raise FileNotFoundError(f"Missing BankChurners CSV at: {path}")

    df = pd.read_csv(path)

    if "Attrition_Flag" not in df.columns:
        raise ValueError("BankChurners dataset must contain 'Attrition_Flag' column.")

    y = (df["Attrition_Flag"].astype(str).str.strip() == "Attrited Customer").astype(int).to_numpy()

    X = df.drop(columns=["Attrition_Flag"], errors="ignore")
    X = _drop_id_cols(X)
    X = _add_engineered_features("bankchurners", X)

    pre = _build_preprocessor_ohe(X)
    return DatasetBundle(name="bankchurners", X=X, y=y, preprocessor=pre)


def load_dataset(name: str) -> DatasetBundle:
    name = name.strip().lower()
    if name == "telco":
        return load_telco()
    if name == "saas":
        return load_saas_csv()
    if name in ("bankchurners", "bank"):
        return load_bankchurners()
    raise ValueError("dataset must be one of: telco, saas, bankchurners")
