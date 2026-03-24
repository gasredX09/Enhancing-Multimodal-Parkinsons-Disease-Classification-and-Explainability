from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_handwriting_svm_embeddings import build_feature_table, load_merged_timeseries

SEED = 42


class OptionalModelError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Benchmark handwriting models, export embeddings, and select best model"
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
        default=repo_root / "outputs" / "unimodal_handwriting" / "model_benchmark",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "f1", "accuracy"],
        help="Metric used to choose best model.",
    )
    return parser.parse_args()


def get_model_factories(seed: int) -> Dict[str, Callable[[], object]]:
    factories: Dict[str, Callable[[], object]] = {
        "svm_rbf": lambda: SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
        "logreg_elasticnet": lambda: LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=8000,
            random_state=seed,
        ),
        "gpc_rbf": lambda: GaussianProcessClassifier(
            kernel=1.0 * RBF(length_scale=1.0),
            random_state=seed,
            max_iter_predict=100,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": lambda: build_xgboost(seed),
        "lightgbm": lambda: build_lightgbm(seed),
        "catboost": lambda: build_catboost(seed),
    }
    return factories


def build_xgboost(seed: int):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise OptionalModelError(f"xgboost unavailable: {exc}") from exc

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def build_lightgbm(seed: int):
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise OptionalModelError(f"lightgbm unavailable: {exc}") from exc

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        class_weight="balanced",
        force_col_wise=True,
        verbosity=-1,
    )


def build_catboost(seed: int):
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        raise OptionalModelError(f"catboost unavailable: {exc}") from exc

    return CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        auto_class_weights="Balanced",
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
    }


def maybe_predict_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        # Squash to pseudo-probability for AUC ranking compatibility.
        return 1.0 / (1.0 + np.exp(-decision))
    pred = model.predict(x)
    return pred.astype(np.float32)


def rank_with_tiebreak(df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    ranked = df.copy()
    # Primary descending, then smaller std preferred.
    ranked = ranked.sort_values(
        by=[f"mean_{primary_metric}", f"std_{primary_metric}", "mean_f1", "mean_accuracy"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def run_model_cv(
    model_name: str,
    model_factory: Callable[[], object],
    X_df: pd.DataFrame,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    skf: StratifiedKFold,
    emb_dim: int,
    model_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict] = []
    oof_parts: List[pd.DataFrame] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_df, y), start=1):
        X_tr_df, X_va_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_df.values)
        X_va_s = scaler.transform(X_va_df.values)

        pca = PCA(n_components=emb_dim, random_state=SEED)
        X_tr_emb = pca.fit_transform(X_tr_s)
        X_va_emb = pca.transform(X_va_s)

        model = model_factory()

        # Keep tree ensembles on raw summary features; others on PCA embeddings.
        use_raw = model_name in {"random_forest", "xgboost", "lightgbm", "catboost"}
        X_fit_tr = X_tr_df if use_raw else X_tr_emb
        X_fit_va = X_va_df if use_raw else X_va_emb

        model.fit(X_fit_tr, y_tr)
        y_pred = model.predict(X_fit_va)
        y_proba = maybe_predict_proba(model, X_fit_va)

        m = metric_dict(y_va, y_pred, y_proba)
        fold_rows.append({
            "model": model_name,
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            **m,
        })

        # Embedding export convention: always save PCA embedding for consistency.
        out = meta_df.iloc[va_idx][["drawing_id", "subject_id", "label"]].copy().reset_index(drop=True)
        for i in range(X_va_emb.shape[1]):
            out[f"emb_{i+1}"] = X_va_emb[:, i]
        out["oof_pred"] = y_pred
        out["oof_proba"] = y_proba
        out["fold"] = fold
        out["model"] = model_name
        oof_parts.append(out)

    metrics_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_parts, ignore_index=True)

    metrics_df.to_csv(model_dir / f"{model_name}_cv_metrics.csv", index=False)
    oof_df.to_csv(model_dir / f"{model_name}_oof_embeddings.csv", index=False)

    return metrics_df, oof_df


def fit_and_save_full_model(
    model_name: str,
    model_factory: Callable[[], object],
    X_df: pd.DataFrame,
    y: np.ndarray,
    emb_dim: int,
    model_dir: Path,
) -> None:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_df.values)

    pca = PCA(n_components=emb_dim, random_state=SEED)
    X_emb = pca.fit_transform(X_s)

    model = model_factory()
    use_raw = model_name in {"random_forest", "xgboost", "lightgbm", "catboost"}
    X_fit = X_df if use_raw else X_emb
    model.fit(X_fit, y)

    artifact = {
        "model_name": model_name,
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "use_raw_features": use_raw,
    }
    joblib.dump(artifact, model_dir / f"{model_name}_full_model.joblib")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading merged data: {args.merged_csv}")
    merged_df = load_merged_timeseries(args.merged_csv)

    print(f"[INFO] Building labeled feature table with metadata: {args.pahaw_meta_xlsx}")
    model_df, feature_cols = build_feature_table(merged_df, args.pahaw_meta_xlsx)
    model_df.to_csv(args.output_dir / "handwriting_summary_features_labeled.csv", index=False)

    X_df = model_df[feature_cols].copy()
    y = model_df["label"].to_numpy(dtype=np.int64)

    emb_dim = min(max(1, args.embedding_dim), X_df.shape[1])
    print(f"[INFO] n_samples={len(model_df)} | n_features={X_df.shape[1]} | emb_dim={emb_dim}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    factories = get_model_factories(args.seed)

    leaderboard_rows: List[Dict] = []
    model_status: Dict[str, str] = {}

    for model_name, factory in factories.items():
        model_dir = args.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Running model: {model_name}")
        try:
            metrics_df, _ = run_model_cv(
                model_name=model_name,
                model_factory=factory,
                X_df=X_df,
                y=y,
                meta_df=model_df,
                skf=skf,
                emb_dim=emb_dim,
                model_dir=model_dir,
            )

            fit_and_save_full_model(
                model_name=model_name,
                model_factory=factory,
                X_df=X_df,
                y=y,
                emb_dim=emb_dim,
                model_dir=model_dir,
            )

            leaderboard_rows.append(
                {
                    "model": model_name,
                    "mean_accuracy": float(metrics_df["accuracy"].mean()),
                    "std_accuracy": float(metrics_df["accuracy"].std(ddof=0)),
                    "mean_f1": float(metrics_df["f1"].mean()),
                    "std_f1": float(metrics_df["f1"].std(ddof=0)),
                    "mean_roc_auc": float(metrics_df["roc_auc"].mean()),
                    "std_roc_auc": float(metrics_df["roc_auc"].std(ddof=0)),
                    "n_splits": int(args.n_splits),
                    "status": "ok",
                }
            )
            model_status[model_name] = "ok"
        except OptionalModelError as exc:
            print(f"[WARN] Skipping optional model {model_name}: {exc}")
            leaderboard_rows.append(
                {
                    "model": model_name,
                    "mean_accuracy": np.nan,
                    "std_accuracy": np.nan,
                    "mean_f1": np.nan,
                    "std_f1": np.nan,
                    "mean_roc_auc": np.nan,
                    "std_roc_auc": np.nan,
                    "n_splits": int(args.n_splits),
                    "status": f"skipped: {exc}",
                }
            )
            model_status[model_name] = f"skipped: {exc}"
        except Exception as exc:
            print(f"[WARN] Model {model_name} failed: {exc}")
            leaderboard_rows.append(
                {
                    "model": model_name,
                    "mean_accuracy": np.nan,
                    "std_accuracy": np.nan,
                    "mean_f1": np.nan,
                    "std_f1": np.nan,
                    "mean_roc_auc": np.nan,
                    "std_roc_auc": np.nan,
                    "n_splits": int(args.n_splits),
                    "status": f"failed: {exc}",
                }
            )
            model_status[model_name] = f"failed: {exc}"

    leaderboard = pd.DataFrame(leaderboard_rows)
    ok_df = leaderboard[leaderboard["status"] == "ok"].copy()

    if ok_df.empty:
        raise RuntimeError("All models failed or were skipped. No benchmark results available.")

    ranked = rank_with_tiebreak(ok_df, primary_metric=args.primary_metric)
    best_model = str(ranked.iloc[0]["model"])

    leaderboard_out = leaderboard.merge(
        ranked[["model", "rank"]], on="model", how="left"
    ).sort_values(by=["rank", "model"], na_position="last")

    leaderboard_path = args.output_dir / "leaderboard.csv"
    leaderboard_out.to_csv(leaderboard_path, index=False)

    best_summary = {
        "primary_metric": args.primary_metric,
        "best_model": best_model,
        "embedding_dim": int(emb_dim),
        "n_splits": int(args.n_splits),
        "feature_columns": feature_cols,
        "n_samples": int(len(model_df)),
        "class_counts": {
            "negative": int((model_df["label"] == 0).sum()),
            "positive": int((model_df["label"] == 1).sum()),
        },
        "model_status": model_status,
        "leaderboard_file": str(leaderboard_path),
    }

    with (args.output_dir / "best_model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    print("\n[RESULT] Top models:")
    print(ranked[["rank", "model", "mean_roc_auc", "mean_f1", "mean_accuracy"]].head(5).to_string(index=False))
    print(f"\n[RESULT] Best model ({args.primary_metric}): {best_model}")
    print(f"[INFO] Leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
