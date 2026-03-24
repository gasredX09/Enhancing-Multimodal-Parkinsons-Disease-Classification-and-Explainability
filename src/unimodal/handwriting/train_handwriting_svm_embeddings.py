from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(
        description="Generate handwriting embeddings with SVM and 5-fold stratified CV"
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=repo_root / "data" / "handwriting" / "processed" / "merged_handwriting_timeseries.csv",
        help="Path to merged handwriting time-series CSV produced by notebook pipeline",
    )
    parser.add_argument(
        "--pahaw-meta-xlsx",
        type=Path,
        default=repo_root / "data" / "handwriting" / "PaHaw Dataset" / "PaHaW_files" / "corpus_PaHaW.xlsx",
        help="Path to PaHaW metadata Excel file for labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "unimodal_handwriting" / "svm_embeddings",
        help="Directory to save embeddings, CV metrics, and model artifacts",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=4,
        help="Target PCA embedding dimension before SVM",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="SVM C regularization parameter",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="SVM gamma parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed",
    )

    return parser.parse_args()


def load_merged_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Merged CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    required = {"drawing_id", "timestamp", "x", "y", "pressure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Merged CSV missing required columns: {sorted(missing)}")

    for col in ["drawing_id", "subject_id", "dataset"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def quality_check(df: pd.DataFrame, min_points: int = 50) -> bool:
    if len(df) < min_points:
        return False
    if df["x"].isna().any() or df["y"].isna().any():
        return False

    x_range = df["x"].max() - df["x"].min()
    y_range = df["y"].max() - df["y"].min()
    return bool(x_range != 0 and y_range != 0)


def extract_summary_features(df: pd.DataFrame) -> pd.Series:
    d = df.sort_values("timestamp").copy()

    d["vx"] = d["x"].diff()
    d["vy"] = d["y"].diff()
    d["speed"] = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2)

    feat = {
        "mean_speed": float(d["speed"].mean()),
        "std_speed": float(d["speed"].std()),
        "max_speed": float(d["speed"].max()),
        "path_length": float(d["speed"].sum()),
        "pressure_mean": float(d["pressure"].mean()),
        "pressure_std": float(d["pressure"].std()),
        "num_points": int(len(d)),
    }
    return pd.Series(feat)


def get_subject_id(drawing_id: str) -> str:
    sid = str(drawing_id).split("__")[0]
    sid = sid.split("_")[0]
    return sid


def uci_label_from_drawing_id(drawing_id: str) -> float:
    d = str(drawing_id)
    if d.startswith("C_"):
        return 0.0
    if d.startswith("P_") or d.startswith("H_"):
        return 1.0
    return np.nan


def build_feature_table(merged_df: pd.DataFrame, meta_xlsx: Path) -> tuple[pd.DataFrame, List[str]]:
    merged_df = merged_df.copy()
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)

    # Remove duplicate timestamps within each drawing.
    dup_ts = merged_df.duplicated(subset=["drawing_id", "timestamp"])
    merged_df = merged_df[~dup_ts].reset_index(drop=True)

    clean_rows: List[pd.DataFrame] = []
    for drawing_id, grp in merged_df.groupby("drawing_id"):
        if quality_check(grp):
            clean_rows.append(grp)

    if not clean_rows:
        raise ValueError("No valid drawings after quality filtering.")

    final_df = pd.concat(clean_rows, ignore_index=True)

    feature_rows: List[pd.Series] = []
    for drawing_id, grp in final_df.groupby("drawing_id"):
        row = extract_summary_features(grp)
        row["drawing_id"] = drawing_id
        feature_rows.append(row)

    summary_df = pd.DataFrame(feature_rows)
    summary_df["subject_id"] = summary_df["drawing_id"].apply(get_subject_id)

    if not meta_xlsx.exists():
        raise FileNotFoundError(f"PaHaW metadata file not found: {meta_xlsx}")

    meta_df = pd.read_excel(meta_xlsx)
    if "ID" not in meta_df.columns or "Disease" not in meta_df.columns:
        raise ValueError("PaHaW metadata must include 'ID' and 'Disease' columns.")

    meta_df["ID"] = meta_df["ID"].astype(str).str.zfill(5)
    meta_df["pa_haw_label"] = meta_df["Disease"].apply(lambda x: 1.0 if x == "PD" else 0.0)

    summary_df = summary_df.merge(
        meta_df[["ID", "pa_haw_label"]],
        left_on="subject_id",
        right_on="ID",
        how="left",
    )

    summary_df["uci_label"] = summary_df["drawing_id"].apply(uci_label_from_drawing_id)
    summary_df["label"] = summary_df["pa_haw_label"].fillna(summary_df["uci_label"])

    feature_cols = [
        "mean_speed",
        "std_speed",
        "max_speed",
        "path_length",
        "pressure_mean",
        "pressure_std",
    ]

    model_df = summary_df.dropna(subset=["label"]).copy()
    if model_df.empty:
        raise ValueError("No labeled examples were found after metadata/ID merge.")

    model_df["label"] = model_df["label"].astype(int)
    return model_df, feature_cols


def run_cv_with_embeddings(model_df: pd.DataFrame, feature_cols: List[str], args: argparse.Namespace) -> Dict:
    X = model_df[feature_cols].to_numpy(dtype=np.float32)
    y = model_df["label"].to_numpy(dtype=np.int64)

    emb_dim = min(max(1, args.embedding_dim), X.shape[1])

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_metrics: List[Dict] = []
    oof_parts: List[pd.DataFrame] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=emb_dim, random_state=args.seed)),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        C=args.c,
                        gamma=args.gamma,
                        probability=True,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_va)
        y_proba = pipe.predict_proba(X_va)[:, 1]

        auc = roc_auc_score(y_va, y_proba) if len(np.unique(y_va)) > 1 else float("nan")
        fold_metrics.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "accuracy": float(accuracy_score(y_va, y_pred)),
                "f1": float(f1_score(y_va, y_pred, zero_division=0)),
                "roc_auc": float(auc),
            }
        )

        scaler = pipe.named_steps["scaler"]
        pca = pipe.named_steps["pca"]
        X_va_emb = pca.transform(scaler.transform(X_va))

        out = model_df.iloc[va_idx][["drawing_id", "subject_id", "label"]].copy().reset_index(drop=True)
        for i in range(X_va_emb.shape[1]):
            out[f"emb_{i+1}"] = X_va_emb[:, i]
        out["oof_pred"] = y_pred
        out["oof_proba"] = y_proba
        out["fold"] = fold
        oof_parts.append(out)

    oof_df = pd.concat(oof_parts, ignore_index=True)

    metric_df = pd.DataFrame(fold_metrics)
    summary = {
        "n_samples": int(len(model_df)),
        "n_positive": int(model_df["label"].sum()),
        "n_negative": int((model_df["label"] == 0).sum()),
        "embedding_dim": int(emb_dim),
        "n_splits": int(args.n_splits),
        "svm": {
            "kernel": "rbf",
            "class_weight": "balanced",
            "C": args.c,
            "gamma": args.gamma,
        },
        "metrics_mean": {
            "accuracy": float(metric_df["accuracy"].mean()),
            "f1": float(metric_df["f1"].mean()),
            "roc_auc": float(metric_df["roc_auc"].mean()),
        },
        "metrics_std": {
            "accuracy": float(metric_df["accuracy"].std(ddof=0)),
            "f1": float(metric_df["f1"].std(ddof=0)),
            "roc_auc": float(metric_df["roc_auc"].std(ddof=0)),
        },
    }

    return {
        "oof_df": oof_df,
        "metrics_df": metric_df,
        "summary": summary,
        "emb_dim": emb_dim,
        "X": X,
        "y": y,
    }


def fit_full_model(X: np.ndarray, y: np.ndarray, emb_dim: int, args: argparse.Namespace) -> Pipeline:
    full_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=emb_dim, random_state=args.seed)),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    C=args.c,
                    gamma=args.gamma,
                    probability=True,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    full_pipe.fit(X, y)
    return full_pipe


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading merged timeseries from: {args.merged_csv}")
    merged_df = load_merged_timeseries(args.merged_csv)

    print(f"[INFO] Building feature table with metadata: {args.pahaw_meta_xlsx}")
    model_df, feature_cols = build_feature_table(merged_df, args.pahaw_meta_xlsx)

    print("[INFO] Running stratified 5-fold SVM CV and generating OOF embeddings")
    results = run_cv_with_embeddings(model_df, feature_cols, args)

    oof_path = args.output_dir / "handwriting_svm_oof_embeddings.csv"
    metrics_path = args.output_dir / "handwriting_svm_cv_metrics.csv"
    summary_path = args.output_dir / "handwriting_svm_cv_summary.json"
    features_path = args.output_dir / "handwriting_summary_features_labeled.csv"

    results["oof_df"].to_csv(oof_path, index=False)
    results["metrics_df"].to_csv(metrics_path, index=False)
    model_df.to_csv(features_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2)

    print("[INFO] Fitting full model on all labeled data")
    full_model = fit_full_model(results["X"], results["y"], results["emb_dim"], args)
    model_path = args.output_dir / "handwriting_svm_full_model.joblib"
    joblib.dump(full_model, model_path)

    print("\n[RESULT] 5-fold CV mean +- std")
    mm = results["summary"]["metrics_mean"]
    ss = results["summary"]["metrics_std"]
    print(f"  accuracy: {mm['accuracy']:.4f} +- {ss['accuracy']:.4f}")
    print(f"  f1      : {mm['f1']:.4f} +- {ss['f1']:.4f}")
    print(f"  roc_auc : {mm['roc_auc']:.4f} +- {ss['roc_auc']:.4f}")

    print("\n[INFO] Saved artifacts:")
    print(f"  - {oof_path}")
    print(f"  - {metrics_path}")
    print(f"  - {summary_path}")
    print(f"  - {features_path}")
    print(f"  - {model_path}")


if __name__ == "__main__":
    main()
