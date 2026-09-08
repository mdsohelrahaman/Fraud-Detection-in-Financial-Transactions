"""Data loading and preprocessing utilities for fraud detection."""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_transactions(path: str = "data/raw/transactions.csv") -> pd.DataFrame:
    """Load the anonymized transaction CSV."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("The transaction dataset is empty.")
    return df


def split_features_target(df: pd.DataFrame, target: str = "is_fraud") -> Tuple[pd.DataFrame, pd.Series]:
    """Split a labeled dataset into features and target."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric and categorical columns."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], remainder="drop")
