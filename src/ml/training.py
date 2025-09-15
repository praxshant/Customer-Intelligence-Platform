# src/ml/training.py
"""
Production training pipeline for CICOP.
- Loads data from the existing DatabaseManager (SQLite by default, can be adapted to Postgres).
- Builds a simple supervised model (sklearn GradientBoosting) for demonstration.
- Saves the trained model artifact under models/model.pkl
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

from src.data.database_manager import DatabaseManager

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


@dataclass
class TrainResult:
    model_path: str
    metrics: Dict[str, Any]


def load_data() -> pd.DataFrame:
    """Load raw transactional dataset joined with customers and segments."""
    db = DatabaseManager()
    df = db.get_transaction_data()
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a simple binary classification target and features for demo purposes.
    Target: whether transaction amount is above customer median amount.
    Features: amount, category frequency, simple recency proxy.
    """
    if df.empty:
        raise ValueError("No data available to train the model. Please generate or load data first.")

    # Basic preprocessing
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    # Compute per-customer median amount for target threshold
    med = df.groupby("customer_id")["amount"].median().rename("cust_median")
    df = df.merge(med, left_on="customer_id", right_index=True, how="left")

    # Target: large transaction indicator
    df["target"] = (df["amount"] > df["cust_median"].fillna(df["amount"].median())).astype(int)

    # Feature: category frequency per customer
    cat_freq = (
        df.groupby(["customer_id", "category"]).size().rename("cat_count").reset_index()
    )
    df = df.merge(cat_freq, on=["customer_id", "category"], how="left")

    # Feature: recency in days from the max date
    max_date = df["transaction_date"].max()
    df["recency_days"] = (max_date - df["transaction_date"]).dt.days.fillna(0)

    # Select features
    feature_cols = ["amount", "cat_count", "recency_days"]
    df["cat_count"].fillna(0, inplace=True)

    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    return X, y


def train_and_save(random_state: int = 42) -> TrainResult:
    """Train model and persist artifact to models/model.pkl"""
    df = load_data()
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        y_prob = None
        auc = None
    acc = float(accuracy_score(y_test, y_pred))

    # Persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return TrainResult(
        model_path=MODEL_PATH,
        metrics={"accuracy": acc, "auc": auc},
    )


def main():
    result = train_and_save()
    print({"saved": result.model_path, "metrics": result.metrics})


if __name__ == "__main__":
    main()
