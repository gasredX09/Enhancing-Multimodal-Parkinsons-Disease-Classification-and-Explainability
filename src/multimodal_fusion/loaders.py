"""
Loaders for unimodal OOF predictions and embeddings.
Use load_all_from_embeddings() as the main entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"


def _find_embedding_file(pattern: str) -> Path:
    """Return the lexicographically last file matching *pattern* in EMBEDDINGS_DIR.

    Also searches for files matching the pattern stem without an extension
    (e.g. 'handwriting_embeddings_0414' alongside '*.csv').
    """
    matches = sorted(EMBEDDINGS_DIR.glob(pattern))
    # Also check for extension-less files whose name matches the stem pattern
    stem_pattern = (
        pattern.rsplit(".*", 1)[0] if ".*" in pattern
        else pattern.replace(".csv", "").replace(".npz", "")
    )
    no_ext = [p for p in EMBEDDINGS_DIR.glob(stem_pattern) if not p.suffix]
    all_matches = sorted(set(matches) | set(no_ext), key=lambda p: p.name)
    if not all_matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {EMBEDDINGS_DIR}.\n"
            "Place a dated embedding file (e.g. gait_embeddings_MMDD.csv) there first."
        )
    return all_matches[-1]


def _fix_handwriting_subject_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix a data bug where PaHaW subjects have subject_id='H' instead of their
    real ID.  Each affected row has a unique drawing_id that serves as the
    correct subject identifier.
    """
    broken = df["subject_id"] == "H"
    if broken.any():
        df = df.copy()
        df.loc[broken, "subject_id"] = df.loc[broken, "drawing_id"]
    return df


def _aggregate_drawings(df: pd.DataFrame, prob_col: str, label_col: str) -> pd.DataFrame:
    """Aggregate drawing-level predictions to subject level by mean probability."""
    return (
        df.groupby("subject_id")
        .agg(y_true=(label_col, "first"), p_pd=(prob_col, "mean"))
        .reset_index()
    )


def load_gait_from_embeddings() -> dict:
    """
    Load gait embeddings from embeddings/.
    NPZ files with y_proba are returned directly; CSV files return X_emb for CV.
    """
    # Prefer NPZ (OOF predictions) over CSV (raw embeddings) when available
    npz_matches = sorted(EMBEDDINGS_DIR.glob("gait_embeddings_*.npz"))
    csv_matches = sorted(EMBEDDINGS_DIR.glob("gait_embeddings_*.csv"))

    # Pick the latest file across both extensions
    all_matches = sorted(npz_matches + csv_matches, key=lambda p: p.name)
    if not all_matches:
        raise FileNotFoundError(
            f"No gait embedding file (gait_embeddings_*.npz or *.csv) found in {EMBEDDINGS_DIR}."
        )
    path = all_matches[-1]

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        # Format B: OOF predictions (keys include y_proba)
        if "y_proba" in keys:
            return {
                "subject_ids": data["subject_ids"].astype(str),
                "y_true":      data["y_true"].astype(int),
                "y_proba":     data["y_proba"].astype(float),
                "task":        "diagnosis",
                "note": (
                    f"Gait OOF predictions. "
                    f"Source: {path.name}. N={len(data['y_true'])} subjects."
                ),
            }
        # Format A: raw embeddings (keys include X_emb)
        label_key = "y_true" if "y_true" in keys else "y"
        feature_names = data["feature_names"].astype(str) if "feature_names" in keys else None
        return {
            "subject_ids":   data["subject_ids"].astype(str),
            "y_true":        data[label_key].astype(int),
            "X_emb":         data["X_emb"].astype(float),
            "feature_names": feature_names,
            "task":          "diagnosis",
            "note": (
                f"Gait raw embeddings. "
                f"Source: {path.name}. N={len(data[label_key])} subjects."
            ),
        }

    # CSV: raw embeddings
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "y")]
    return {
        "subject_ids":   df["subject_id"].astype(str).values,
        "y_true":        df["y"].astype(int).values,
        "X_emb":         df[feature_cols].values.astype(float),
        "feature_names": np.array(feature_cols),
        "task":          "diagnosis",
        "note": (
            "WearGait subject-level concatenated embeddings "
            f"(SelfPace + HurriedPace + TUG). "
            f"Source: {path.name}. N={len(df)} subjects."
        ),
    }


def load_handwriting_from_embeddings(clf=None) -> dict:
    """
    Load handwriting embeddings from embeddings/.
    Accepts either a raw embedding CSV (runs OOF CV via clf) or a pre-computed OOF prediction file.
    """
    csv_path = _find_embedding_file("handwriting_embeddings_0406.csv")
    df = pd.read_csv(csv_path)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    is_embedding_format = bool(emb_cols) and "oof_proba" not in df.columns

    if is_embedding_format:
        # Format B: raw embeddings — derive subject_id from drawing_id
        df["subject_id"] = df["drawing_id"].str.split("__").str[0]
        agg = (
            df.groupby("subject_id")
            .agg(
                y_true=("label", "first"),
                **{c: (c, "mean") for c in emb_cols},
            )
            .reset_index()
        )
        emb_dict = {
            "subject_ids": agg["subject_id"].astype(str).values,
            "y_true": agg["y_true"].astype(int).values,
            "X_emb": agg[emb_cols].values.astype(float),
            "feature_names": np.array(emb_cols),
            "task": "diagnosis",
            "note": f"Handwriting raw embeddings. Source: {csv_path.name}.",
        }
        clf_name = type(clf).__name__ if clf is not None else "LogisticRegression"
        pred_dict = compute_oof_predictions(emb_dict, clf=clf)
        pred_dict["note"] = (
            f"Handwriting OOF predictions via 5-fold {clf_name} CV on {len(emb_cols)}-dim "
            f"embeddings. Source: {csv_path.name}. N={len(agg)} subjects."
        )
        return pred_dict

    # Format A: OOF prediction file
    df = _fix_handwriting_subject_ids(df)
    agg = _aggregate_drawings(df, prob_col="oof_proba", label_col="label")
    p_pd = agg["p_pd"].values.astype(float)
    model_name = df["model"].iloc[0] if "model" in df.columns else "unknown"
    return {
        "subject_ids": agg["subject_id"].astype(str).values,
        "y_true": agg["y_true"].astype(int).values,
        "y_proba": np.column_stack([1.0 - p_pd, p_pd]),
        "task": "diagnosis",
        "note": (
            f"Handwriting OOF predictions (model={model_name}, "
            f"drawing->subject mean). "
            f"Source: {csv_path.name}. N={len(agg)} subjects."
        ),
    }


def load_speech_from_embeddings(use_emb: str = "concat", clf=None) -> dict:
    """
    Load speech embeddings from embeddings/.
    use_emb: 'concat' (194-dim, default), 'cb_proba' (used directly), 'cnn', or 'cb_leaf'.
    """
    npz_path = _find_embedding_file("speech_embeddings_*.npz")
    data = np.load(npz_path, allow_pickle=True)
    subject_ids = data["participant_id"].astype(str)
    y_true = data["y"].astype(int)

    key_map = {
        "cb_proba": "emb_cb_proba",
        "cnn": "emb_cnn",
        "cb_leaf": "emb_cb_leaf",
        "concat": "emb_concat",
    }
    if use_emb not in key_map:
        raise ValueError(
            f"Unknown use_emb='{use_emb}'. Choose from: {list(key_map)}."
        )
    arr = data[key_map[use_emb]].astype(float)

    if use_emb == "cb_proba":
        y_proba = arr  # already (N, 2)
    else:
        emb_dict = {
            "subject_ids": subject_ids,
            "y_true": y_true,
            "X_emb": arr,
            "feature_names": None,
            "task": "diagnosis",
            "note": "",
        }
        y_proba = compute_oof_predictions(emb_dict, clf=clf)["y_proba"]

    clf_name = type(clf).__name__ if clf is not None else "LogisticRegression"
    return {
        "subject_ids": subject_ids,
        "y_true": y_true,
        "y_proba": y_proba,
        "task": "diagnosis",
        "note": (
            f"Speech predictions (use_emb={use_emb}, clf={clf_name}). "
            f"Source: {npz_path.name}. N={len(subject_ids)} participants."
        ),
    }


def compute_oof_predictions(
    emb_dict: dict,
    clf=None,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Run stratified k-fold CV on emb_dict['X_emb'] and return OOF probabilities."""
    x_emb = emb_dict["X_emb"].astype(float)
    y = emb_dict["y_true"].astype(int)
    subject_ids = emb_dict["subject_ids"]

    if clf is None:
        clf = LogisticRegression(max_iter=1000, C=0.1, random_state=random_state)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    y_proba = np.zeros((len(y), 2), dtype=float)

    for train_idx, val_idx in skf.split(x_emb, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_emb[train_idx])
        x_val = scaler.transform(x_emb[val_idx])
        fold_clf = type(clf)(**clf.get_params())
        fold_clf.fit(x_train, y[train_idx])
        y_proba[val_idx] = fold_clf.predict_proba(x_val)

    return {
        "subject_ids": subject_ids,
        "y_true": y,
        "y_proba": y_proba,
        "task": emb_dict.get("task", "diagnosis"),
        "note": (
            f"OOF predictions via {cv}-fold CV ({type(clf).__name__}). "
            + emb_dict.get("note", "")
        ),
    }


def load_all_from_embeddings(
    speech_use_emb: str = "concat",
    include_gait: bool = True,
    clf=None,
) -> dict[str, dict]:
    # Load all modalities from embeddings/, returning OOF prediction dicts for LateFusionModel.fit().
    modalities: dict[str, dict] = {}

    if include_gait:
        try:
            gait_data = load_gait_from_embeddings()
            # NPZ format already contains y_proba; CSV format has X_emb and needs CV
            if "y_proba" in gait_data:
                modalities["gait"] = gait_data
            else:
                modalities["gait"] = compute_oof_predictions(gait_data, clf=clf)
        except FileNotFoundError as exc:
            print(f"[WARNING] Skipping gait embeddings - {exc}")

    try:
        modalities["handwriting"] = load_handwriting_from_embeddings(clf=clf)
    except FileNotFoundError as exc:
        print(f"[WARNING] Skipping handwriting embeddings - {exc}")

    try:
        modalities["speech"] = load_speech_from_embeddings(use_emb=speech_use_emb, clf=clf)
    except FileNotFoundError as exc:
        print(f"[WARNING] Skipping speech embeddings - {exc}")

    return modalities
