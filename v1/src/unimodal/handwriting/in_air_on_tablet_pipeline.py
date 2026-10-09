import sys
from pathlib import Path
from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

sys.path.append(str(Path(__file__).parent))

from in_air_handwriting_pipeline_2 import (
    extract_in_air_features,
    get_subject_id,
    uci_label_from_drawing_id,
)
from train_handwriting_svm_embeddings import extract_summary_features

SEED = 42


def try_import_xgboost(seed: int):
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
    except Exception as e:
        print(f"Could not import xgboost: {e}")
        return None


def try_import_lightgbm(seed: int):
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
            class_weight="balanced",
            force_col_wise=True,
            random_state=seed,
            verbosity=-1,
        )
    except Exception as e:
        print(f"Could not import lightgbm: {e}")
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


def needs_scaling(model_name: str) -> bool:
    return model_name in {"svm_rbf", "logistic_regression", "gpc_rbf"}


def safe_stats(arr: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def extract_kinematic_features(
    grp: pd.DataFrame,
    pressure_mode: str,
    prefix: str,
    low_speed_quantile: float = 0.1,
) -> pd.Series:
    """
    pressure_mode:
      - 'in_air'     -> pressure == 0
      - 'on_tablet'  -> pressure > 0
    """
    d = grp.sort_values("timestamp").copy()

    if pressure_mode == "in_air":
        sub = d[d["pressure"] == 0].copy()
    elif pressure_mode == "on_tablet":
        sub = d[d["pressure"] > 0].copy()
    else:
        raise ValueError("pressure_mode must be 'in_air' or 'on_tablet'")

    if len(sub) < 3:
        out = {
            f"{prefix}_acc_mean": 0.0,
            f"{prefix}_acc_std": 0.0,
            f"{prefix}_acc_max": 0.0,
            f"{prefix}_jerk_mean": 0.0,
            f"{prefix}_jerk_std": 0.0,
            f"{prefix}_jerk_max": 0.0,
            f"{prefix}_pause_count": 0.0,
            f"{prefix}_pause_ratio": 0.0,
            f"{prefix}_turn_mean": 0.0,
            f"{prefix}_turn_std": 0.0,
        }
        return pd.Series(out)

    sub["dx"] = sub["x"].diff()
    sub["dy"] = sub["y"].diff()
    sub["dt"] = sub["timestamp"].diff()

    step = sub.dropna(subset=["dx", "dy", "dt"]).copy()
    step = step[step["dt"] > 0].copy()

    if len(step) < 2:
        out = {
            f"{prefix}_acc_mean": 0.0,
            f"{prefix}_acc_std": 0.0,
            f"{prefix}_acc_max": 0.0,
            f"{prefix}_jerk_mean": 0.0,
            f"{prefix}_jerk_std": 0.0,
            f"{prefix}_jerk_max": 0.0,
            f"{prefix}_pause_count": 0.0,
            f"{prefix}_pause_ratio": 0.0,
            f"{prefix}_turn_mean": 0.0,
            f"{prefix}_turn_std": 0.0,
        }
        return pd.Series(out)

    step["dist"] = np.sqrt(step["dx"] ** 2 + step["dy"] ** 2)
    step["speed"] = step["dist"] / step["dt"]

    # Acceleration
    step["prev_speed"] = step["speed"].shift(1)
    step["acc"] = (step["speed"] - step["prev_speed"]) / step["dt"]
    acc = step["acc"].dropna().to_numpy()

    # Jerk
    step["prev_acc"] = step["acc"].shift(1)
    step["jerk"] = (step["acc"] - step["prev_acc"]) / step["dt"]
    jerk = step["jerk"].dropna().to_numpy()

    # Pause features from low-speed segments
    speed_vals = step["speed"].to_numpy()
    dt_vals = step["dt"].to_numpy()
    valid_speed = speed_vals[np.isfinite(speed_vals)]
    if valid_speed.size == 0:
        pause_count = 0.0
        pause_ratio = 0.0
    else:
        threshold = np.quantile(valid_speed, low_speed_quantile)
        pause_mask = speed_vals <= threshold
        pause_count = float(np.sum((pause_mask[1:] & ~pause_mask[:-1]).astype(int)) + (1 if pause_mask[0] else 0))
        total_time = float(np.sum(dt_vals)) if np.sum(dt_vals) > 0 else 0.0
        pause_time = float(np.sum(dt_vals[pause_mask]))
        pause_ratio = pause_time / total_time if total_time > 0 else 0.0

    # Turning-angle features
    vx = step["dx"].to_numpy()
    vy = step["dy"].to_numpy()
    norms = np.sqrt(vx**2 + vy**2)
    turn_angles = []
    for i in range(1, len(vx)):
        if norms[i - 1] == 0 or norms[i] == 0:
            continue
        cosang = (vx[i - 1] * vx[i] + vy[i - 1] * vy[i]) / (norms[i - 1] * norms[i])
        cosang = np.clip(cosang, -1.0, 1.0)
        turn_angles.append(np.arccos(cosang))
    turn_angles = np.asarray(turn_angles, dtype=float)

    out = {}
    out.update(safe_stats(np.abs(acc), f"{prefix}_acc"))
    out.update(safe_stats(np.abs(jerk), f"{prefix}_jerk"))
    out[f"{prefix}_pause_count"] = pause_count
    out[f"{prefix}_pause_ratio"] = pause_ratio
    out[f"{prefix}_turn_mean"] = float(np.mean(turn_angles)) if turn_angles.size else 0.0
    out[f"{prefix}_turn_std"] = float(np.std(turn_angles)) if turn_angles.size else 0.0

    return pd.Series(out)


def build_feature_tables(df: pd.DataFrame) -> pd.DataFrame:
    in_air_rows = []
    tablet_rows = []

    for drawing_id, grp in df.groupby("drawing_id"):
        # Existing in-air features
        row_in_air = extract_in_air_features(grp)

        for col in ["in_air_num_points", "in_air_points_per_sec"]:
            if col in row_in_air:
                del row_in_air[col]

        # New in-air kinematic features
        in_air_kin = extract_kinematic_features(grp, pressure_mode="in_air", prefix="in_air")
        for k, v in in_air_kin.items():
            row_in_air[k] = v

        row_in_air["drawing_id"] = drawing_id
        if "dataset" in grp.columns:
            row_in_air["dataset"] = grp["dataset"].iloc[0]
        in_air_rows.append(row_in_air)

        # Existing on-tablet summary features
        row_tab = extract_summary_features(grp)
        if "num_points" in row_tab:
            del row_tab["num_points"]

        # New on-tablet kinematic features
        tablet_kin = extract_kinematic_features(grp, pressure_mode="on_tablet", prefix="tablet")
        for k, v in tablet_kin.items():
            row_tab[k] = v

        row_tab["drawing_id"] = drawing_id
        if "dataset" in grp.columns:
            row_tab["dataset"] = grp["dataset"].iloc[0]
        tablet_rows.append(row_tab)

    in_air_df = pd.DataFrame(in_air_rows)
    tablet_df = pd.DataFrame(tablet_rows)

    if "dataset" in tablet_df.columns:
        tablet_df = tablet_df.drop(columns=["dataset"])

    features = pd.merge(
        in_air_df,
        tablet_df,
        on="drawing_id",
        how="inner",
        suffixes=("_inair", "_ontablet"),
    )
    return features


def add_labels(features: pd.DataFrame, meta_xlsx: Path) -> pd.DataFrame:
    features = features.copy()
    features["subject_id"] = features["drawing_id"].apply(get_subject_id)
    features["uci_label"] = features["drawing_id"].apply(uci_label_from_drawing_id)

    meta_df = pd.read_excel(meta_xlsx, engine="openpyxl")
    meta_df["ID"] = meta_df["ID"].astype(str).str.zfill(5)
    meta_df["pahaw_label"] = meta_df["Disease"].apply(
        lambda x: 1 if str(x).strip().upper() == "PD" else 0
    )

    features = features.merge(
        meta_df[["ID", "pahaw_label"]],
        left_on="subject_id",
        right_on="ID",
        how="left",
    )

    features["label"] = features["pahaw_label"].fillna(features["uci_label"])
    features = features.dropna(subset=["label"]).copy()
    features["label"] = features["label"].astype(int)
    return features


def get_feature_columns(features: pd.DataFrame):
    # Only keep the requested features (user-specified)
    requested_features = [
        "in_air_time",
        "in_air_total_distance",
        "in_air_mean_speed",
        "in_air_std_speed",
        "in_air_num_segments",
        "in_air_pause_count",
        "in_air_pause_ratio",
        "in_air_turn_mean",
        "in_air_turn_std",
        "in_air_acc_mean",
        "in_air_jerk_mean",
        "mean_speed",
        "std_speed",
        "path_length",
        "pressure_mean",
        "pressure_std",
        "tablet_acc_mean",
        "tablet_pause_ratio",
        "tablet_turn_mean",
    ]
    feature_cols = [c for c in requested_features if c in features.columns]
    features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    features[feature_cols] = features[feature_cols].fillna(0)
    return features, feature_cols


def get_model_factories() -> Dict[str, Callable[[], Optional[object]]]:
    return {
        "svm_rbf": lambda: SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=SEED,
        ),
        "logistic_regression": lambda: LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=8000,
            random_state=SEED,
        ),
        "gpc_rbf": lambda: GaussianProcessClassifier(
            kernel=1.0 * RBF(length_scale=1.0),
            random_state=SEED,
            max_iter_predict=100,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        # "xgboost" and "lightgbm" excluded
    }


def main():
    merged_csv = Path(
        "/ocean/projects/med260006p/mkhelgi/biomedAI/project/data/handwriting/processed/merged_handwriting_timeseries.csv"
    )
    meta_xlsx = Path(
        "/ocean/projects/med260006p/mkhelgi/biomedAI/project/data/handwriting/PaHaw Dataset/PaHaW_files/corpus_PaHaW.xlsx"
    )
    output_dir = Path(
        "/ocean/projects/med260006p/mkhelgi/biomedAI/project/outputs/unimodal_handwriting/in_air_on_tablet_acc_jerk"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    n_splits = 5

    print(f"Reading merged CSV from: {merged_csv}")
    df = pd.read_csv(merged_csv, low_memory=False)

    print("\nDataset counts:")
    if "dataset" in df.columns:
        print(df["dataset"].value_counts(dropna=False))

    features = build_feature_tables(df)
    features = add_labels(features, meta_xlsx)
    features, feature_cols = get_feature_columns(features)

    print("\nNumber of drawings in final feature table:", len(features))
    print("Number of feature columns:", len(feature_cols))
    print("\nFeature columns:")
    print(feature_cols)

    print("\nLabel counts:")
    print(features["label"].value_counts(dropna=False))

    if "dataset" in features.columns:
        print("\nLabel distribution by dataset:")
        print(pd.crosstab(features["dataset"], features["label"]))

    features_out = output_dir / "in_air_on_tablet_features_with_acc_jerk.csv"
    features.to_csv(features_out, index=False)
    print(f"\nSaved feature table to: {features_out}")

    X = features[feature_cols].to_numpy(dtype=np.float32)
    y = features["label"].to_numpy(dtype=np.int64)

    # --- Out-of-fold PCA Embedding Export for Best Model (random_forest) ---
    # Embeddings will be saved to: outputs/unimodal_handwriting/in_air_on_tablet_acc_jerk/embeddings_random_forest.csv
    from sklearn.decomposition import PCA
    embedding_dim = min(8, len(feature_cols))  # You can adjust the embedding dimension as needed
    oof_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        X_train = X[train_idx]
        X_test = X[test_idx]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=embedding_dim, random_state=SEED)
        X_train_emb = pca.fit_transform(X_train_scaled)
        X_test_emb = pca.transform(X_test_scaled)
        for idx, emb in zip(test_idx, X_test_emb):
            row = {
                "drawing_id": features.iloc[idx]["drawing_id"],
                "label": features.iloc[idx]["label"],
                "fold": fold,
            }
            for i, val in enumerate(emb):
                row[f"emb_{i+1}"] = val
            oof_rows.append(row)
    emb_df = pd.DataFrame(oof_rows)
    emb_out = output_dir / "embeddings_random_forest.csv"
    emb_df.to_csv(emb_out, index=False)
    print(f"\nSaved out-of-fold PCA embeddings for best model (random_forest) to: {emb_out}")

    model_factories = get_model_factories()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    all_results = []


    # --- Model training and embedding export ---
    embedding_dim = 4  # You can adjust this as needed
    best_model_name = "random_forest"  # Set to best model
    embedding_rows = []

    for model_name, factory in model_factories.items():
        print(f"\nRunning {model_name}...")
        if factory() is None:
            print(f"{model_name} unavailable, skipping.")
            continue

        fold_results = []

        # For embedding export, collect out-of-fold embeddings for best model
        if model_name == best_model_name:
            from sklearn.decomposition import PCA
            oof_embeddings = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if needs_scaling(model_name):
                scaler = StandardScaler()
                X_train_use = scaler.fit_transform(X_train)
                X_test_use = scaler.transform(X_test)
            else:
                X_train_use = X_train
                X_test_use = X_test

            model = factory()
            if model is None:
                continue

            model.fit(X_train_use, y_train)
            y_pred = model.predict(X_test_use)
            y_proba = maybe_predict_proba(model, X_test_use)

            m = metric_dict(y_test, y_pred, y_proba)
            m["fold"] = fold
            m["model"] = model_name
            fold_results.append(m)
            all_results.append(m)

            # --- Embedding export for best model ---
            if model_name == best_model_name:
                # Fit PCA on training set, transform test set
                pca = PCA(n_components=min(embedding_dim, X_train.shape[1]), random_state=SEED)
                X_train_scaled = X_train_use
                X_test_scaled = X_test_use
                pca.fit(X_train_scaled)
                X_test_emb = pca.transform(X_test_scaled)
                # Save drawing_id, label, fold, and embeddings
                for idx, emb in zip(test_idx, X_test_emb):
                    row = {
                        "drawing_id": features.iloc[idx]["drawing_id"],
                        "label": features.iloc[idx]["label"],
                        "fold": fold,
                    }
                    for i, val in enumerate(emb):
                        row[f"emb_{i+1}"] = val
                    embedding_rows.append(row)

        df_results = pd.DataFrame(fold_results)
        out_path = output_dir / f"cv_results_{model_name}.csv"
        df_results.to_csv(out_path, index=False)
        print(df_results)
        print(f"\n{model_name} mean results:")
        print(df_results[["accuracy", "f1", "roc_auc"]].mean())

    # Save embeddings for best model
    if embedding_rows:
        emb_df = pd.DataFrame(embedding_rows)
        emb_out = output_dir / f"embeddings_{best_model_name}.csv"
        emb_df.to_csv(emb_out, index=False)
        print(f"\nSaved embeddings for {best_model_name} to: {emb_out}")

    all_results_df = pd.DataFrame(all_results)
    all_results_out = output_dir / "cv_results_all_models.csv"
    all_results_df.to_csv(all_results_out, index=False)

    summary_df = (
        all_results_df.groupby("model")[["accuracy", "f1", "roc_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "model",
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std",
        "roc_auc_mean", "roc_auc_std",
    ]

    summary_out = output_dir / "summary_all_models.csv"
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved all fold results to: {all_results_out}")
    print(f"Saved summary to: {summary_out}")
    print("\nSummary:")
    print(summary_df.sort_values("roc_auc_mean", ascending=False))


if __name__ == "__main__":
    main()