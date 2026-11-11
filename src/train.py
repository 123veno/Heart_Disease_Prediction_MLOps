"""Reproducible training and evaluation for the heart-disease demo."""
import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FEATURES, TARGET, CATEGORICAL_FEATURES, NUMERIC_FEATURES, COLUMN_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heartdisease.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

def load_data(path=DATA_PATH):
    # Cleveland-style file has no header; '?' denotes unknown ca/thal values.
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values=["?"])
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    # Original target: 0 = no disease; 1..4 = disease present.
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df

def build_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args()

    for folder in [MODEL_DIR, REPORT_DIR, LOG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR/"training.log", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = load_data(args.data)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear", "lbfgs"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, param_grid, scoring="recall", cv=cv, n_jobs=-1,
        refit=True, error_score="raise"
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    pred = best_model.predict(X_test)
    prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "best_params": search.best_params_,
        "cv_best_recall": float(search.best_score_),
        "dataset": "UCI Cleveland Heart Disease (local CSV supplied by user)",
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "target_mapping": "original target 0 -> 0; original targets 1,2,3,4 -> 1"
    }

    joblib.dump(best_model, MODEL_DIR/"heart_disease_pipeline.joblib")
    with open(MODEL_DIR/"model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "features": FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "target_mapping": metrics["target_mapping"],
            "model_type": "LogisticRegression with preprocessing Pipeline"
        }, f, indent=2)

    with open(REPORT_DIR/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(REPORT_DIR/"classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, pred, target_names=["No disease", "Disease"]))

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No disease", "Disease"],
                yticklabels=["No disease", "Disease"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title("ROC Curve - Logistic Regression")
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"roc_curve.png", dpi=160)
    plt.close()

    # Optional MLflow tracking. Training still works when MLflow is not installed.
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("heart-disease-logistic-regression")
        with mlflow.start_run():
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_param("dataset_rows", len(df))
            mlflow.log_param("dataset_file", Path(args.data).name)
            mlflow.log_artifact(
                str(MODEL_DIR / "heart_disease_pipeline.joblib"),
                artifact_path="model"
            )
            mlflow.log_artifact(str(REPORT_DIR / "metrics.json"))
            # mlflow.sklearn.log_model(best_model, artifact_path="model")
            # mlflow.log_artifact(str(REPORT_DIR/"metrics.json"))
    except ImportError:
        logging.info("MLflow not installed; skipped experiment tracking.")
    except Exception as exc:
        logging.exception("MLflow logging failed: %s", exc)

    logging.info("Training completed. Metrics: %s", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_DIR/'heart_disease_pipeline.joblib'}")

if __name__ == "__main__":
    main()
