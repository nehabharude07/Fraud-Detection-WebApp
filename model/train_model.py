"""
Train a fraud-detection model on the onlinefraud.csv dataset and save it
(along with feature metadata) for the Flask app to load at runtime.

The full dataset has ~6.36M rows with heavy class imbalance (~0.13% fraud).
To keep training fast and reproducible we take a stratified sample that
keeps every fraud case and a random slice of the legitimate cases.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "onlinefraud.csv"
MODEL_OUT = "fraud_model.joblib"
META_OUT = "model_meta.json"

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["errorBalanceOrig"] = (
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    type_dummies = pd.get_dummies(df["type"], drop_first=True)
    for col in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        if col not in type_dummies:
            type_dummies[col] = 0
    df = pd.concat([df, type_dummies], axis=1)
    return df


def main():
    t0 = time.time()
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0].sample(n=200_000, random_state=42)
    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample):,} rows ({len(fraud):,} fraud, {len(legit):,} legit)")

    sample = engineer_features(sample)

    X = sample[FEATURE_COLUMNS]
    y = sample["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_name, best_model, best_auc = None, None, -1
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, output_dict=True)
        results[name] = {"auc": auc, "accuracy": acc, "confusion_matrix": cm, "report": report}
        print(f"  AUC={auc:.4f}  Accuracy={acc:.4f}")
        if auc > best_auc:
            best_name, best_model, best_auc = name, model, auc

    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")

    joblib.dump(best_model, MODEL_OUT)

    meta = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "transaction_types": TRANSACTION_TYPES,
        "metrics": {k: {"auc": v["auc"], "accuracy": v["accuracy"]} for k, v in results.items()},
        "full_results": results,
        "trained_rows": len(sample),
        "fraud_rows": int(len(fraud)),
        "training_seconds": round(time.time() - t0, 1),
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {MODEL_OUT} and {META_OUT} in {meta['training_seconds']}s")


if __name__ == "__main__":
    main()
