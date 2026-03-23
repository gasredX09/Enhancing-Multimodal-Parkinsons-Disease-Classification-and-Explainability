from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_handwriting_svm_embeddings import build_feature_table, load_merged_timeseries

SEED = 42


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(
        description="Finalize handwriting model with nested CV, calibration, and threshold selection"
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=repo_root / "data" / "handwriting" / "processed" / "merged_handwriting_timeseries.csv",
    )
    parser.add_argument(
        "--pahaw-meta-xlsx",
        type=Path,
        default=repo_root / "data" / "handwriting" / "PaHaw Dataset" / "PaHaW_files" / "corpus_PaHaW.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "unimodal_handwriting" / "final_model",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--primary-metric", type=str, default="roc_auc", choices=["roc_auc", "f1", "accuracy"])
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def get_model_spaces(seed: int) -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    spaces: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {
        "gpc_rbf": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=seed)),
                ]
            ),
            {
                "model__kernel": [
                    1.0 * RBF(length_scale=0.5),
                    1.0 * RBF(length_scale=1.0),
                    1.0 * RBF(length_scale=2.0),
                ]
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6, None],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
    }

    # Prefer LightGBM if installed; otherwise fallback to elastic-net logistic.
    try:
        from lightgbm import LGBMClassifier

        spaces["lightgbm"] = (
            LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=seed,
                force_col_wise=True,
                verbosity=-1,
            ),
            {
                "n_estimators": [150, 300],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [7, 15, 31],
                "min_child_samples": [5, 10],
            },
        )
    except Exception:
        spaces["logreg_elasticnet"] = (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight="balanced",
                            max_iter=10000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
        )

    return spaces


def fit_tuned_model(
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    inner_splits: int,
    seed: int,
) -> Any:
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )
    gs.fit(x_train, y_train)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def threshold_by_recall(y_true: np.ndarray, y_proba: np.ndarray, target_recall: float) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_prec = -1.0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall and prec > best_prec:
            best_prec = prec
            best_thr = float(thr)

    return best_thr


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_proba >= thr).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["roc_auc"] = float("nan")
    return out


def rank_models(results_df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    ranked = results_df.sort_values(
        by=[f"mean_{primary_metric}", f"std_{primary_metric}", "mean_f1", "mean_accuracy"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading merged data: {args.merged_csv}")
    merged_df = load_merged_timeseries(args.merged_csv)

    print(f"[INFO] Building feature table with metadata: {args.pahaw_meta_xlsx}")
    model_df, feature_cols = build_feature_table(merged_df, args.pahaw_meta_xlsx)

    x = model_df[feature_cols].copy()
    y = model_df["label"].to_numpy(dtype=np.int64)

    spaces = get_model_spaces(args.seed)
    outer_cv = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.seed)

    leaderboard_rows: List[Dict[str, Any]] = []
    all_oof_parts: List[pd.DataFrame] = []

    for model_name, (estimator, param_grid) in spaces.items():
        print(f"[INFO] Nested-CV for model: {model_name}")

        fold_metrics: List[Dict[str, Any]] = []
        best_params_per_fold: List[Dict[str, Any]] = []
        fold_proba = np.zeros(len(model_df), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(x, y), start=1):
            x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            best_estimator, best_params, best_inner_auc = fit_tuned_model(
                estimator=estimator,
                param_grid=param_grid,
                x_train=x_tr,
                y_train=y_tr,
                inner_splits=args.inner_splits,
                seed=args.seed,
            )

            # Calibrate probabilities on training partition only.
            calibrator = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=3)
            calibrator.fit(x_tr, y_tr)

            y_va_proba = calibrator.predict_proba(x_va)[:, 1]
            fold_proba[va_idx] = y_va_proba

            fold_thr = threshold_by_recall(y_tr, calibrator.predict_proba(x_tr)[:, 1], args.target_recall)
            m = compute_metrics(y_va, y_va_proba, fold_thr)
            m["fold"] = fold
            m["best_inner_auc"] = best_inner_auc
            fold_metrics.append(m)
            best_params_per_fold.append({"fold": fold, "params": best_params})

        # Global threshold over full OOF probabilities.
        global_thr = threshold_by_recall(y, fold_proba, args.target_recall)
        overall = compute_metrics(y, fold_proba, global_thr)

        fold_df = pd.DataFrame(fold_metrics)
        row = {
            "model": model_name,
            "mean_accuracy": float(fold_df["accuracy"].mean()),
            "std_accuracy": float(fold_df["accuracy"].std(ddof=0)),
            "mean_precision": float(fold_df["precision"].mean()),
            "std_precision": float(fold_df["precision"].std(ddof=0)),
            "mean_recall": float(fold_df["recall"].mean()),
            "std_recall": float(fold_df["recall"].std(ddof=0)),
            "mean_f1": float(fold_df["f1"].mean()),
            "std_f1": float(fold_df["f1"].std(ddof=0)),
            "mean_roc_auc": float(fold_df["roc_auc"].mean()),
            "std_roc_auc": float(fold_df["roc_auc"].std(ddof=0)),
            "oof_threshold": float(global_thr),
            "oof_accuracy": overall["accuracy"],
            "oof_precision": overall["precision"],
            "oof_recall": overall["recall"],
            "oof_f1": overall["f1"],
            "oof_roc_auc": overall["roc_auc"],
        }
        leaderboard_rows.append(row)

        per_model_dir = args.output_dir / model_name
        per_model_dir.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(per_model_dir / "nested_cv_fold_metrics.csv", index=False)

        with (per_model_dir / "best_params_per_fold.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(best_params_per_fold), f, indent=2)

        oof_df = model_df[["drawing_id", "subject_id", "label"]].copy()
        oof_df["oof_proba"] = fold_proba
        oof_df["oof_pred"] = (fold_proba >= global_thr).astype(int)
        oof_df["model"] = model_name
        oof_df.to_csv(per_model_dir / "oof_predictions.csv", index=False)
        all_oof_parts.append(oof_df)

    leaderboard = pd.DataFrame(leaderboard_rows)
    ranked = rank_models(leaderboard, args.primary_metric)
    best_model_name = str(ranked.iloc[0]["model"])

    leaderboard_path = args.output_dir / "final_model_leaderboard.csv"
    ranked.to_csv(leaderboard_path, index=False)

    all_oof = pd.concat(all_oof_parts, ignore_index=True)
    all_oof.to_csv(args.output_dir / "all_models_oof_predictions.csv", index=False)

    # Fit best model on full data with inner-grid search, then calibrate on full data.
    best_estimator, best_params, best_cv_auc = fit_tuned_model(
        estimator=spaces[best_model_name][0],
        param_grid=spaces[best_model_name][1],
        x_train=x,
        y_train=y,
        inner_splits=args.inner_splits,
        seed=args.seed,
    )
    best_calibrated = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=5)
    best_calibrated.fit(x, y)

    chosen_thr = float(ranked.iloc[0]["oof_threshold"])

    artifact = {
        "model_name": best_model_name,
        "feature_columns": feature_cols,
        "model": best_calibrated,
        "threshold": chosen_thr,
        "target_recall": args.target_recall,
        "primary_metric": args.primary_metric,
        "best_params": best_params,
        "best_inner_cv_auc": best_cv_auc,
    }
    model_artifact_path = args.output_dir / "final_handwriting_model.joblib"
    joblib.dump(artifact, model_artifact_path)

    summary = {
        "best_model": best_model_name,
        "primary_metric": args.primary_metric,
        "target_recall": args.target_recall,
        "threshold": chosen_thr,
        "n_samples": int(len(model_df)),
        "class_counts": {
            "negative": int((model_df["label"] == 0).sum()),
            "positive": int((model_df["label"] == 1).sum()),
        },
        "feature_columns": feature_cols,
        "best_params": to_jsonable(best_params),
        "best_inner_cv_auc": best_cv_auc,
        "leaderboard_file": str(leaderboard_path),
        "artifact_file": str(model_artifact_path),
    }

    with (args.output_dir / "final_model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    model_df.to_csv(args.output_dir / "final_model_input_table.csv", index=False)

    print("\n[RESULT] Final ranked models")
    print(ranked[["rank", "model", "mean_roc_auc", "mean_f1", "mean_accuracy", "oof_threshold"]].to_string(index=False))
    print(f"\n[RESULT] Selected model: {best_model_name}")
    print(f"[RESULT] Threshold: {chosen_thr:.3f} (target recall={args.target_recall:.2f})")
    print(f"[INFO] Leaderboard: {leaderboard_path}")
    print(f"[INFO] Artifact: {model_artifact_path}")


if __name__ == "__main__":
    main()
