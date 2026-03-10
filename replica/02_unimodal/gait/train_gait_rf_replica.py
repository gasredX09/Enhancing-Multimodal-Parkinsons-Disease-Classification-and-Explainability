"""
Recreate the original unimodal gait tabular baseline style without copying code.

Pipeline:
1) Load engineered gait features from features.csv
2) Train/evaluate Random Forest with StratifiedGroupKFold (5 folds)
3) Train/evaluate Random Forest with Leave-One-Subject-Out
4) Save CSV outputs in the same spirit as the original repository artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold


SEED = 42


def build_default_paths() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    source_features = (
        repo_root
        / "Multimodal-Parkinson-Disease-Prediction-With-XAI"
        / "parkinson"
        / "features.csv"
    )
    output_dir = repo_root / "project" / "replica" / "outputs" / "unimodal_gait_rf_replica"
    return {"features": source_features, "output": output_dir}


def parse_args() -> argparse.Namespace:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Replicate original RF gait baselines from features.csv")
    parser.add_argument("--features-csv", type=Path, default=defaults["features"], help="Path to features.csv")
    parser.add_argument("--output-dir", type=Path, default=defaults["output"], help="Directory for CSV outputs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=500, help="RF trees")
    return parser.parse_args()


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"features.csv not found: {path}")

    df = pd.read_csv(path)
    required = {"subject", "trial", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("features.csv is empty")

    return df


def make_xyg(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in {"subject", "trial", "label"}]
    X = df[feat_cols].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()
    groups = df["subject"].astype(str).to_numpy()
    return X, y, groups, feat_cols


def build_rf(seed: int, n_estimators: int) -> RandomForestClassifier:
    # Balanced class weights mirror the original handling for class imbalance.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    uniq = np.unique(y_true)
    if uniq.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def run_sgkf(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_estimators: int) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: List[Dict] = []

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        model = build_rf(seed=seed, n_estimators=n_estimators)
        model.fit(X[tr], y[tr])

        y_pred = model.predict(X[te])
        y_prob = model.predict_proba(X[te])[:, 1]

        rows.append(
            {
                "fold": fold,
                "n_test": int(len(te)),
                "acc": float(accuracy_score(y[te], y_pred)),
                "f1": float(f1_score(y[te], y_pred, zero_division=0)),
                "auc": safe_auc(y[te], y_prob),
            }
        )

    return pd.DataFrame(rows)


def run_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_estimators: int) -> pd.DataFrame:
    logo = LeaveOneGroupOut()
    rows: List[Dict] = []

    for tr, te in logo.split(X, y, groups):
        held_out = str(groups[te][0])
        model = build_rf(seed=seed, n_estimators=n_estimators)
        model.fit(X[tr], y[tr])

        y_true = y[te]
        y_pred = model.predict(X[te])
        classes_in_test = np.unique(y_true).tolist()

        rows.append(
            {
                "held_out": held_out,
                "n_test": int(len(te)),
                "acc": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "classes_in_test": str(classes_in_test),
            }
        )

    return pd.DataFrame(rows).sort_values("held_out").reset_index(drop=True)


def summarize(df_sgkf: pd.DataFrame, df_loso: pd.DataFrame, n_subjects: int, n_samples: int, n_features: int) -> Dict:
    return {
        "sgkf": {
            "folds": int(len(df_sgkf)),
            "mean_acc": float(df_sgkf["acc"].mean()),
            "mean_f1": float(df_sgkf["f1"].mean()),
            "mean_auc": float(df_sgkf["auc"].mean()),
            "std_acc": float(df_sgkf["acc"].std(ddof=0)),
            "std_f1": float(df_sgkf["f1"].std(ddof=0)),
            "std_auc": float(df_sgkf["auc"].std(ddof=0)),
        },
        "loso": {
            "held_out_subjects": int(len(df_loso)),
            "mean_acc": float(df_loso["acc"].mean()),
            "mean_f1": float(df_loso["f1"].mean()),
            "std_acc": float(df_loso["acc"].std(ddof=0)),
            "std_f1": float(df_loso["f1"].std(ddof=0)),
        },
        "dataset": {
            "n_samples": int(n_samples),
            "n_subjects": int(n_subjects),
            "n_features": int(n_features),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(args.features_csv)
    X, y, groups, feat_cols = make_xyg(df)

    df_sgkf = run_sgkf(X, y, groups, seed=args.seed, n_estimators=args.n_estimators)
    df_loso = run_loso(X, y, groups, seed=args.seed, n_estimators=args.n_estimators)

    out_sgkf = args.output_dir / "rf_sgkf_per_fold.csv"
    out_loso = args.output_dir / "rf_loso_per_fold.csv"
    out_summary = args.output_dir / "summary.json"

    df_sgkf.to_csv(out_sgkf, index=False)
    df_loso.to_csv(out_loso, index=False)

    summary = summarize(
        df_sgkf=df_sgkf,
        df_loso=df_loso,
        n_subjects=df["subject"].nunique(),
        n_samples=len(df),
        n_features=len(feat_cols),
    )
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {out_sgkf}")
    print(f"  {out_loso}")
    print(f"  {out_summary}")
    print("\nSGKF means:")
    print(df_sgkf[["acc", "f1", "auc"]].mean().to_string())
    print("\nLOSO means:")
    print(df_loso[["acc", "f1"]].mean().to_string())


if __name__ == "__main__":
    main()
