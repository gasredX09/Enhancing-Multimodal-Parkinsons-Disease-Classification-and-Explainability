"""
Handwriting SVM + Embedding Generator
======================================
Extracts hand-crafted kinematic features from the merged handwriting timeseries,
trains a balanced SVM classifier, and saves subject-level embeddings (scaled
feature vectors + predicted probabilities) for downstream multimodal fusion.

Outputs
-------
src/unimodal/handwriting/processed/
    handwriting_embeddings.csv   - one row per subject, features + label + SVM proba
    handwriting_svm_scaler.pkl   - fitted RobustScaler
    handwriting_svm_model.pkl    - fitted SVC (probability=True)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]          # project root
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
TIMESERIES_CSV = PROCESSED_DIR / "merged_handwriting_timeseries.csv"
PAHAW_META_XLSX = (
    BASE_DIR
    / "replica"
    / "data"
    / "handwriting"
    / "PaHaw Dataset"
    / "PaHaW_files"
    / "corpus_PaHaW.xlsx"
)

OUT_EMBEDDINGS = PROCESSED_DIR / "handwriting_embeddings.csv"
OUT_SCALER     = PROCESSED_DIR / "handwriting_svm_scaler.pkl"
OUT_MODEL      = PROCESSED_DIR / "handwriting_svm_model.pkl"

FEATURE_COLS = [
    "mean_speed",
    "std_speed",
    "max_speed",
    "path_length",
    "mean_accel",
    "std_accel",
    "mean_jerk",
    "pressure_mean",
    "pressure_std",
    "pressure_range",
    # num_points intentionally excluded: reflects recording duration / tablet
    # sample rate, not motor impairment — a dataset-identity confound.
]

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _load_pahaw_meta(xlsx_path: Path) -> dict:
    """Return {zero-padded-ID: label} for PaHaW subjects (0=healthy, 1=PD)."""
    meta = pd.read_excel(xlsx_path)
    meta["ID"] = meta["ID"].astype(str).str.zfill(5)
    label_map = {
        row["ID"]: (1 if row["Disease"] == "PD" else 0)
        for _, row in meta.iterrows()
    }
    return label_map


def get_label(drawing_id: str, pahaw_map: dict) -> int | None:
    """
    Assign binary label:
        0 = healthy control
        1 = Parkinson's disease

    UCI naming conventions
    ----------------------
    C_xxxx  → healthy control
    H_xxxx  → PD patient (UCI parkinson folder)
    P_xxxx  → PD patient (UCI parkinson folder)

    PaHaW naming convention
    -----------------------
    00016_00016__1_1  → subject_id is the leading 5-digit zero-padded integer
    """
    d = str(drawing_id)

    # UCI subjects
    if d.startswith("C_"):
        return 0
    if d.startswith(("H_", "P_")):
        return 1

    # PaHaW subjects – extract numeric ID
    subject_id = d.split("__")[0].split("_")[0].zfill(5)
    return pahaw_map.get(subject_id, None)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(group: pd.DataFrame) -> pd.Series:
    """Compute kinematic summary features from one drawing's timeseries."""
    g = group.sort_values("timestamp").copy()

    # First-order spatial differences → speed
    g["vx"] = g["x"].diff()
    g["vy"] = g["y"].diff()
    g["speed"] = np.sqrt(g["vx"] ** 2 + g["vy"] ** 2)

    # Second-order → acceleration
    g["ax"] = g["vx"].diff()
    g["ay"] = g["vy"].diff()
    g["accel"] = np.sqrt(g["ax"] ** 2 + g["ay"] ** 2)

    # Third-order → jerk
    g["jx"] = g["ax"].diff()
    g["jy"] = g["ay"].diff()
    g["jerk"] = np.sqrt(g["jx"] ** 2 + g["jy"] ** 2)

    return pd.Series(
        {
            "mean_speed":     g["speed"].mean(),
            "std_speed":      g["speed"].std(),
            "max_speed":      g["speed"].max(),
            "path_length":    g["speed"].sum(),
            "mean_accel":     g["accel"].mean(),
            "std_accel":      g["accel"].std(),
            "mean_jerk":      g["jerk"].mean(),
            "pressure_mean":  g["pressure"].mean(),
            "pressure_std":   g["pressure"].std(),
            "pressure_range": g["pressure"].max() - g["pressure"].min(),
            "num_points":     float(len(g)),
        }
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("Loading timeseries …")
    df = pd.read_csv(TIMESERIES_CSV, low_memory=False)
    print(f"  {len(df):,} rows  |  {df['drawing_id'].nunique()} drawings")

    print("Loading PaHaW metadata …")
    pahaw_map = _load_pahaw_meta(PAHAW_META_XLSX)

    # --- Feature extraction (one row per drawing) --------------------------
    print("Extracting features …")
    feature_rows = []
    for drawing_id, group in df.groupby("drawing_id"):
        feats = extract_features(group)
        feats["drawing_id"] = drawing_id
        feats["subject_id"] = group["subject_id"].iloc[0]   # use original CSV column
        feats["dataset"]    = group["dataset"].iloc[0]
        feature_rows.append(feats)

    summary = pd.DataFrame(feature_rows).reset_index(drop=True)

    # --- Labels ------------------------------------------------------------
    summary["label"] = summary["drawing_id"].apply(
        lambda d: get_label(d, pahaw_map)
    )

    # Drop rows with unknown labels
    before = len(summary)
    summary = summary.dropna(subset=["label"]).reset_index(drop=True).copy()
    summary["label"] = summary["label"].astype(int)
    print(f"  Labelled: {len(summary)}/{before}  |  "
          f"PD={summary['label'].sum()}  Healthy={(summary['label']==0).sum()}")

    # --- Train / test split ------------------------------------------------
    X = summary[FEATURE_COLS].values
    y = summary["label"].values

    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, np.arange(len(summary)),
        test_size=0.2, stratify=y, random_state=42
    )

    # --- Scaling -----------------------------------------------------------
    # RobustScaler uses median/IQR, making it resistant to the extreme
    # pressure_range outliers present in this dataset.
    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- SVM ---------------------------------------------------------------
    print("Training SVM …")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale",
              class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_train_sc, y_train)

    # --- Evaluation --------------------------------------------------------
    y_pred  = svm.predict(X_test_sc)
    y_proba = svm.predict_proba(X_test_sc)[:, 1]

    print("\n-- Test-set performance --------------------------------------")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "PD"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # CV uses a fresh Pipeline so the scaler is re-fit inside each fold,
    # preventing any scaling information from leaking into CV test splits.
    cv_pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                       class_weight="balanced", probability=True, random_state=42)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # --- Build embedding table (all samples) -------------------------------
    print("\nGenerating embeddings for all subjects …")
    X_all_sc = scaler.transform(X)
    proba_all = svm.predict_proba(X_all_sc)[:, 1]

    emb = summary[["drawing_id", "subject_id", "dataset", "label"]].copy()
    for i, col in enumerate(FEATURE_COLS):
        emb[f"feat_{col}"] = X_all_sc[:, i]        # scaled features
    emb["svm_proba_pd"] = proba_all

    # Mark train / test split
    split = pd.array(["train"] * len(summary), dtype=object)
    split[idx_test] = "test"
    emb["split"] = split

    # --- Save outputs ------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    emb.to_csv(OUT_EMBEDDINGS, index=False)
    joblib.dump(scaler, OUT_SCALER)
    joblib.dump(svm,    OUT_MODEL)

    print(f"\nSaved embeddings -> {OUT_EMBEDDINGS}")
    print(f"Saved scaler     -> {OUT_SCALER}")
    print(f"Saved model      -> {OUT_MODEL}")
    print(f"\nEmbedding shape: {emb.shape}  (rows x cols)")
    print(emb.head())


if __name__ == "__main__":
    main()
