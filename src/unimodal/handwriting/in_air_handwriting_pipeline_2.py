import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined UCI + PaHaW in-air handwriting benchmark pipeline"
    )
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
        help="Path to PaHaW metadata Excel file for labels",
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
        "--svm-c",
        type=float,
        default=1.0,
        help="SVM C regularization parameter",
    )
    parser.add_argument(
        "--svm-gamma",
        type=str,
        default="scale",
        help="SVM gamma parameter",
    )
    return parser.parse_args()


def get_subject_id(drawing_id: str) -> str:
    drawing_id = str(drawing_id)
    sid = drawing_id.split("__")[0]
    sid = sid.split("_")[0]
    return sid


def uci_label_from_drawing_id(drawing_id: str) -> float:
    drawing_id = str(drawing_id)
    if drawing_id.startswith("C_"):
        return 0.0
    if drawing_id.startswith("P_") or drawing_id.startswith("H_"):
        return 1.0
    return np.nan


def safe_divide(num: float, den: float) -> float:
    if den is None or den == 0 or np.isnan(den):
        return 0.0
    return float(num / den)


def extract_in_air_features(df: pd.DataFrame) -> pd.Series:
    d = df.sort_values("timestamp").copy()
    in_air = d[d["pressure"] == 0].copy()

    if in_air.empty or len(in_air) < 2:
        return pd.Series({
            "in_air_num_points": 0,
            "in_air_time": 0.0,
            "in_air_total_distance": 0.0,
            "in_air_mean_speed": 0.0,
            "in_air_std_speed": 0.0,
            "in_air_max_speed": 0.0,
            "in_air_num_segments": 0,
            "in_air_points_per_sec": 0.0,
            "in_air_distance_per_sec": 0.0,
            "in_air_segments_per_sec": 0.0,
        })

    in_air["dx"] = in_air["x"].diff()
    in_air["dy"] = in_air["y"].diff()
    in_air["dt"] = in_air["timestamp"].diff()

    step_df = in_air.dropna(subset=["dx", "dy", "dt"]).copy()
    step_df = step_df[step_df["dt"] > 0].copy()

    if step_df.empty:
        return pd.Series({
            "in_air_num_points": int(len(in_air)),
            "in_air_time": 0.0,
            "in_air_total_distance": 0.0,
            "in_air_mean_speed": 0.0,
            "in_air_std_speed": 0.0,
            "in_air_max_speed": 0.0,
            "in_air_num_segments": 1 if len(in_air) > 0 else 0,
            "in_air_points_per_sec": 0.0,
            "in_air_distance_per_sec": 0.0,
            "in_air_segments_per_sec": 0.0,
        })

    step_df["step_distance"] = np.sqrt(step_df["dx"] ** 2 + step_df["dy"] ** 2)
    step_df["speed"] = step_df["step_distance"] / step_df["dt"]

    in_air_num_points = int(len(in_air))
    in_air_time = float(step_df["dt"].sum())
    in_air_total_distance = float(step_df["step_distance"].sum())
    in_air_mean_speed = float(step_df["speed"].mean())
    in_air_std_speed = float(step_df["speed"].std(ddof=0)) if len(step_df) > 1 else 0.0
    in_air_max_speed = float(step_df["speed"].max())

    gap_threshold = 0.1
    num_breaks = int((step_df["dt"] > gap_threshold).sum())
    in_air_num_segments = num_breaks + 1 if in_air_num_points > 0 else 0

    in_air_points_per_sec = safe_divide(in_air_num_points, in_air_time)
    in_air_distance_per_sec = safe_divide(in_air_total_distance, in_air_time)
    in_air_segments_per_sec = safe_divide(in_air_num_segments, in_air_time)

    return pd.Series({
        "in_air_num_points": in_air_num_points,
        "in_air_time": in_air_time,
        "in_air_total_distance": in_air_total_distance,
        "in_air_mean_speed": in_air_mean_speed,
        "in_air_std_speed": in_air_std_speed,
        "in_air_max_speed": in_air_max_speed,
        "in_air_num_segments": in_air_num_segments,
        "in_air_points_per_sec": in_air_points_per_sec,
        "in_air_distance_per_sec": in_air_distance_per_sec,
        "in_air_segments_per_sec": in_air_segments_per_sec,
    })


def build_in_air_feature_table(
    merged_df: pd.DataFrame,
    meta_xlsx: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    merged_df = merged_df.copy()
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)

    dup_ts = merged_df.duplicated(subset=["drawing_id", "timestamp"])
    merged_df = merged_df[~dup_ts].reset_index(drop=True)

    feature_rows = []
    for drawing_id, grp in merged_df.groupby("drawing_id"):
        row = extract_in_air_features(grp)
        row["drawing_id"] = drawing_id
        row["dataset"] = grp["dataset"].iloc[0] if "dataset" in grp.columns else "unknown"
        feature_rows.append(row)

    summary_df = pd.DataFrame(feature_rows)
    summary_df["subject_id"] = summary_df["drawing_id"].apply(get_subject_id)

    meta_df = pd.read_excel(meta_xlsx, engine="openpyxl")
    meta_df["ID"] = meta_df["ID"].astype(str).str.zfill(5)
    meta_df["pa_haw_label"] = meta_df["Disease"].apply(
        lambda x: 1.0 if str(x).strip().upper() == "PD" else 0.0
    )

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
        "in_air_mean_speed",
        "in_air_std_speed",
        "in_air_max_speed",
        "in_air_points_per_sec",
        "in_air_distance_per_sec",
        "in_air_segments_per_sec",
    ]

    model_df = summary_df.dropna(subset=["label"]).copy()
    model_df["label"] = model_df["label"].astype(int)

    model_df[feature_cols] = model_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    model_df[feature_cols] = model_df[feature_cols].fillna(0)

    return model_df, feature_cols


def get_models(args) -> Dict[str, object]:
    models = {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=args.svm_c,
                gamma=args.svm_gamma,
                kernel="rbf",
                probability=True,
                random_state=SEED,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            )),
        ]),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED,
            use_label_encoder=False,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=SEED,
            verbosity=-1,
        ),
        "gpc_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianProcessClassifier(
                kernel=1.0 * RBF(length_scale=1.0),
                random_state=SEED,
            )),
        ]),
    }
    return models


def get_positive_class_scores(model, X_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        scores = np.asarray(scores, dtype=float)
        min_s = scores.min()
        max_s = scores.max()
        if max_s > min_s:
            return (scores - min_s) / (max_s - min_s)
        return np.zeros_like(scores, dtype=float)

    return model.predict(X_test).astype(float)


def run_benchmark_cv(
    model_df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = model_df[feature_cols].to_numpy(dtype=np.float32)
    y = model_df["label"].to_numpy(dtype=np.int64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    models = get_models(args)

    all_results = []

    for model_name, model in models.items():
        print(f"\nRunning model: {model_name}")
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = get_positive_class_scores(model, X_test)

            fold_result = {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
                "fold": fold_idx,
            }
            all_results.append(fold_result)

    results_df = pd.DataFrame(all_results)

    summary_df = (
        results_df.groupby("model")[["accuracy", "f1", "roc_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df.columns = [
        "model",
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std",
        "roc_auc_mean", "roc_auc_std",
    ]

    return results_df, summary_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv, low_memory=False)

    print("Dataset counts from merged CSV:")
    if "dataset" in df.columns:
        print(df["dataset"].value_counts(dropna=False))
    else:
        print("No 'dataset' column found.")

    model_df, feature_cols = build_in_air_feature_table(df, args.meta_xlsx)

    print("\nFeature columns used:")
    print(feature_cols)

    print("\nLabel counts:")
    print(model_df["label"].value_counts(dropna=False))

    if "dataset" in model_df.columns:
        print("\nDrawings per dataset in final model table:")
        print(model_df["dataset"].value_counts(dropna=False))

        print("\nLabel distribution by dataset:")
        print(pd.crosstab(model_df["dataset"], model_df["label"]))

    features_out = args.output_dir / "combined_in_air_features_uci_pahaw.csv"
    model_df.to_csv(features_out, index=False)
    print(f"\nSaved combined feature table to: {features_out}")

    results_df, summary_df = run_benchmark_cv(
        model_df=model_df,
        feature_cols=feature_cols,
        n_splits=args.n_splits,
        args=args,
    )

    results_out = args.output_dir / "in_air_benchmark_cv_results_combined_uci_pahaw.csv"
    summary_out = args.output_dir / "in_air_benchmark_summary_combined_uci_pahaw.csv"

    results_df.to_csv(results_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved fold-level results to: {results_out}")
    print(f"Saved summary results to: {summary_out}")

    print("\nFold-level results:")
    print(results_df)

    print("\nSummary results:")
    print(summary_df.sort_values("roc_auc_mean", ascending=False))


if __name__ == "__main__":
    main()