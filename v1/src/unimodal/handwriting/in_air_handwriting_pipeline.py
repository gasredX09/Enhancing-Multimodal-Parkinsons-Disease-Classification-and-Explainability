import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Handwriting in-air features pipeline")
    parser.add_argument(
        "--merged-csv",
        type=Path,
        required=True,
        help="Path to merged handwriting time-series CSV",
    )
    parser.add_argument(
        "--meta-xlsx",
        type=Path,
        required=True,
        help="Path to metadata Excel file for labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds",
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
    return parser.parse_args()

def extract_in_air_features(df: pd.DataFrame) -> pd.Series:
    d = df.sort_values("timestamp").copy()
    # In-air: pressure == 0
    in_air = d[d["pressure"] == 0].copy()
    if in_air.empty:
        return pd.Series({
            "in_air_time": 0.0,
            "in_air_distance": 0.0,
            "in_air_mean_speed": 0.0,
            "in_air_segments": 0,
        })
    in_air["vx"] = in_air["x"].diff()
    in_air["vy"] = in_air["y"].diff()
    in_air["speed"] = np.sqrt(in_air["vx"] ** 2 + in_air["vy"] ** 2)
    in_air["dt"] = in_air["timestamp"].diff().fillna(0)
    in_air_time = in_air["dt"].sum()
    in_air_distance = in_air["speed"].sum()
    in_air_mean_speed = in_air["speed"].mean() if len(in_air) > 0 else 0.0
    # Count in-air segments (continuous runs)
    in_air_segments = (in_air["dt"] > 0.1).sum() + 1 if len(in_air) > 0 else 0
    return pd.Series({
        "in_air_time": float(in_air_time),
        "in_air_distance": float(in_air_distance),
        "in_air_mean_speed": float(in_air_mean_speed),
        "in_air_segments": int(in_air_segments),
    })

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

def build_in_air_feature_table(merged_df: pd.DataFrame, meta_xlsx: Path) -> Tuple[pd.DataFrame, List[str]]:
    merged_df = merged_df.copy()
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
    dup_ts = merged_df.duplicated(subset=["drawing_id", "timestamp"])
    merged_df = merged_df[~dup_ts].reset_index(drop=True)
    feature_rows = []
    for drawing_id, grp in merged_df.groupby("drawing_id"):
        row = extract_in_air_features(grp)
        row["drawing_id"] = drawing_id
        feature_rows.append(row)
    summary_df = pd.DataFrame(feature_rows)
    summary_df["subject_id"] = summary_df["drawing_id"].apply(get_subject_id)
    meta_df = pd.read_excel(meta_xlsx, engine="openpyxl")
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
        "in_air_time",
        "in_air_distance",
        "in_air_mean_speed",
        "in_air_segments",
    ]
    model_df = summary_df.dropna(subset=["label"]).copy()
    model_df["label"] = model_df["label"].astype(int)
    return model_df, feature_cols

def main():
    
    args = parse_args()
    df = pd.read_csv(args.merged_csv)
    print(df["dataset"].value_counts())
    datasets = df["dataset"].unique()
    for dataset in datasets:
        print(f"\nRunning analysis for dataset: {dataset}")
        df_subset = df[df["dataset"] == dataset].copy()
        model_df, feature_cols = build_in_air_feature_table(df_subset, args.meta_xlsx)
        # Fill any NaNs in feature columns with 0
        model_df[feature_cols] = model_df[feature_cols].fillna(0)
        X = model_df[feature_cols].to_numpy(dtype=np.float32)
        y = model_df["label"].to_numpy(dtype=np.int64)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
        results = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(C=args.c, gamma=args.gamma, probability=True, random_state=SEED)),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            results.append({
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            })
        results_df = pd.DataFrame(results)
        out_file = args.output_dir / f"in_air_cv_results_{dataset}.csv"
        results_df.to_csv(out_file, index=False)
        print(f"Results for {dataset} saved to {out_file}")
        print("Mean results:")
        print(results_df.mean())

if __name__ == "__main__":
    main()
