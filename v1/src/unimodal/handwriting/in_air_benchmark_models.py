import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def try_import_xgboost(seed):
    try:
        from xgboost import XGBClassifier
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
    except Exception:
        return None


def try_import_lightgbm(seed):
    try:
        from lightgbm import LGBMClassifier
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
    except Exception:
        return None


def try_import_catboost(seed):
    try:
        from catboost import CatBoostClassifier
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
    except Exception:
        return None


def metric_dict(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
    }


def maybe_predict_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))
    pred = model.predict(x)
    return pred.astype(np.float32)


def extract_in_air_features(group):
    group = group.sort_values("timestamp").copy()

    if len(group) < 2:
        return None

    x = group["x"].to_numpy(dtype=float)
    y = group["y"].to_numpy(dtype=float)
    t = group["timestamp"].to_numpy(dtype=float)
    p = group["pressure"].to_numpy(dtype=float) if "pressure" in group.columns else np.zeros(len(group))

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    dt_safe = np.where(dt == 0, np.nan, dt)

    dist = np.sqrt(dx**2 + dy**2)
    speed = dist / dt_safe

    in_air_mask = p[:-1] == 0
    in_air_dist = dist[in_air_mask]
    in_air_speed = speed[in_air_mask]

    features = {
        "n_points": len(group),
        "duration": float(t.max() - t.min()) if len(t) > 0 else 0.0,
        "pressure_mean": float(np.nanmean(p)),
        "pressure_std": float(np.nanstd(p)),
        "penup_fraction": float(np.mean(p == 0)),
        "total_path_length": float(np.nansum(dist)),
        "mean_speed": float(np.nanmean(speed)) if len(speed) else 0.0,
        "std_speed": float(np.nanstd(speed)) if len(speed) else 0.0,
        "max_speed": float(np.nanmax(speed)) if len(speed) else 0.0,
        "in_air_path_length": float(np.nansum(in_air_dist)) if len(in_air_dist) else 0.0,
        "in_air_mean_speed": float(np.nanmean(in_air_speed)) if len(in_air_speed) else 0.0,
        "in_air_std_speed": float(np.nanstd(in_air_speed)) if len(in_air_speed) else 0.0,
        "in_air_max_speed": float(np.nanmax(in_air_speed)) if len(in_air_speed) else 0.0,
        "n_penup_segments": int(np.sum((p[:-1] > 0) & (p[1:] == 0))),
    }

    return pd.Series(features)


def build_in_air_feature_table(df):
    required_cols = ["drawing_id", "timestamp", "x", "y", "pressure", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Merged CSV is missing required columns: {missing}")

    feature_rows = []

    for drawing_id, group in df.groupby("drawing_id"):
        feat = extract_in_air_features(group)
        if feat is None:
            continue

        feat["drawing_id"] = drawing_id
        feat["label"] = group["label"].iloc[0]

        if "dataset" in group.columns:
            feat["dataset"] = group["dataset"].iloc[0]
        if "subject_id" in group.columns:
            feat["subject_id"] = group["subject_id"].iloc[0]

        feature_rows.append(feat)

    model_df = pd.DataFrame(feature_rows)

    if model_df.empty:
        raise ValueError("No usable drawings were found after feature extraction.")

    model_df = model_df.dropna(subset=["label"]).copy()
    model_df["label"] = model_df["label"].astype(int)

    feature_cols = [
        c for c in model_df.columns
        if c not in {"drawing_id", "label", "dataset", "subject_id"}
    ]

    return model_df, feature_cols


def run_cv(model_name, model_factory, X, y, n_splits, use_pca, emb_dim, output_dir, dataset_name):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(
                f"[WARNING] Fold {fold} for {model_name} skipped: "
                f"only one class present. y_train={np.unique(y_train)}, y_test={np.unique(y_test)}"
            )
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if use_pca:
            n_components = min(max(1, emb_dim), X_train_s.shape[1], X_train_s.shape[0])
            pca = PCA(n_components=n_components, random_state=42)
            X_train_s = pca.fit_transform(X_train_s)
            X_test_s = pca.transform(X_test_s)

        model = model_factory()
        if model is None:
            print(f"{model_name} not available, skipping.")
            continue

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = maybe_predict_proba(model, X_test_s)

        m = metric_dict(y_test, y_pred, y_proba)
        m["fold"] = fold
        results.append(m)

    results_df = pd.DataFrame(results)

    out_file = output_dir / f"in_air_cv_results_{model_name}_{dataset_name}.csv"
    results_df.to_csv(out_file, index=False)

    if results_df.empty:
        print(f"{model_name} results for {dataset_name}: no valid folds.")
    else:
        print(f"\n{model_name} results for {dataset_name}:")
        print(results_df.mean(numeric_only=True))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark in-air handwriting features with multiple models"
    )
    parser.add_argument("--merged-csv", type=Path, required=True)
    parser.add_argument(
        "--meta-xlsx",
        type=Path,
        required=False,
        help="Optional. Only needed if your pipeline still requires external metadata."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/in_air_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="If set, only run models on this dataset (e.g., PaHaW or UCI)"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv)

    required_cols = {"drawing_id", "timestamp", "x", "y", "pressure", "dataset", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Merged CSV is missing required columns: {sorted(missing_cols)}"
        )

    print("Loaded merged CSV:", args.merged_csv)
    print("Columns:", list(df.columns))
    print("Label distribution:")
    print(df["label"].value_counts(dropna=False))

    # --- Per-dataset loop (commented out for combined run) ---
    # if args.dataset:
    #     datasets = [args.dataset]
    # else:
    #     datasets = sorted(df["dataset"].dropna().unique())
    #
    # for dataset in datasets:
    #     print(f"\nRunning analysis for dataset: {dataset}")
    #     df_subset = df[df["dataset"] == dataset].copy()
    #
    #     model_df, feature_cols = build_in_air_feature_table(df_subset)
    #     model_df[feature_cols] = model_df[feature_cols].fillna(0)
    #
    #     print(f"Feature table shape for {dataset}: {model_df.shape}")
    #     print("Class balance:")
    #     print(model_df["label"].value_counts(dropna=False))
    #
    #     X = model_df[feature_cols].to_numpy(dtype=np.float32)
    #     y = model_df["label"].to_numpy(dtype=np.int64)
    #
    #     emb_dim = min(max(1, args.embedding_dim), X.shape[1])
    #
    #     model_factories = {
    #         ...
    #     }
    #
    #     for model_name, factory in model_factories.items():
    #         use_pca = model_name not in {"random_forest", "xgboost", "lightgbm", "catboost"}
    #         run_cv(
    #             ...
    #         )

    # --- Combined run on all data ---
    print("\nRunning analysis on ALL data combined (no dataset split)")
    model_df, feature_cols = build_in_air_feature_table(df)
    model_df[feature_cols] = model_df[feature_cols].fillna(0)
    print(f"Feature table shape (ALL): {model_df.shape}")
    print("Class balance:")
    print(model_df["label"].value_counts(dropna=False))
    X = model_df[feature_cols].to_numpy(dtype=np.float32)
    y = model_df["label"].to_numpy(dtype=np.int64)
    emb_dim = min(max(1, args.embedding_dim), X.shape[1])
    model_factories = {
        "svm_rbf": lambda: SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        ),
        "logreg_elasticnet": lambda: LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=8000,
            random_state=42
        ),
        "gpc_rbf": lambda: GaussianProcessClassifier(
            kernel=1.0 * RBF(length_scale=1.0),
            random_state=42,
            max_iter_predict=100
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": lambda: try_import_xgboost(42),
        "lightgbm": lambda: try_import_lightgbm(42),
        "catboost": lambda: try_import_catboost(42),
    }
    for model_name, factory in model_factories.items():
        use_pca = model_name not in {"random_forest", "xgboost", "lightgbm", "catboost"}
        run_cv(
            model_name=model_name,
            model_factory=factory,
            X=X,
            y=y,
            n_splits=args.n_splits,
            use_pca=use_pca,
            emb_dim=emb_dim,
            output_dir=args.output_dir,
            dataset_name="ALL",
        )


if __name__ == "__main__":
    main()