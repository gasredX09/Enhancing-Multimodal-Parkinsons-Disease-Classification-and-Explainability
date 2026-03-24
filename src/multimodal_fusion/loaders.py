"""
Standardized loaders for each unimodal model's OOF predictions and embeddings.

Each prediction loader returns a dict with keys:
    subject_ids : np.ndarray of str, shape (N,)
    y_true      : np.ndarray of int, shape (N,)
    y_proba     : np.ndarray of float, shape (N, 2)
    task        : str
    note        : str

Embedding loaders return subject-level features with keys:
    subject_ids : np.ndarray of str, shape (N,)
    y_true      : np.ndarray of int, shape (N,)
    X_emb       : np.ndarray of float, shape (N, D)
    feature_names : np.ndarray[str] or None
    task        : str
    note        : str
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_SRC = Path(__file__).parent.parent
_PROJECT = _SRC.parent
OUTPUTS_ROOT = _PROJECT / "outputs"
SPEECH_ROOT = _SRC / "unimodal" / "speech" / "scripts"


def load_gait_predictions() -> dict:
    npz_path = OUTPUTS_ROOT / "unimodal_gait" / "PDFE_Severity_Classification" / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Gait predictions not found at {npz_path}\n"
            "Run src/unimodal/gait/train_gait.py first."
        )
    data = np.load(npz_path, allow_pickle=True)
    return {
        "subject_ids": data["subject_ids"].astype(str),
        "y_true": data["y_true"].astype(int),
        "y_proba": data["y_proba"].astype(float),
        "task": "severity",
        "note": "PDFE Figshare: PD-only severity classification (mild=0 vs moderate/severe=1). N=35.",
    }


def load_gait_concat_embeddings() -> dict:
    npz_path = (
        OUTPUTS_ROOT
        / "unimodal_gait"
        / "weargait_concat_embeddings"
        / "weargait_concat_subject_embeddings.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Concatenated WearGait embeddings not found at {npz_path}\n"
            "Run src/unimodal/gait/gait_ensemble_orchestrator.py --tasks weargait first."
        )
    data = np.load(npz_path, allow_pickle=True)
    feature_names = data["feature_names"].astype(str) if "feature_names" in data else None
    return {
        "subject_ids": data["subject_ids"].astype(str),
        "y_true": data["y"].astype(int),
        "X_emb": data["X_emb"].astype(float),
        "feature_names": feature_names,
        "task": "diagnosis",
        "note": (
            "WearGait subject-level concatenated embeddings from separate SelfPace, "
            "HurriedPace, and TUG TCN encoders."
        ),
    }


def load_handwriting_predictions(model: str = "svm") -> dict:
    if model == "svm":
        csv_path = (
            OUTPUTS_ROOT
            / "unimodal_handwriting"
            / "svm_embeddings"
            / "handwriting_svm_oof_embeddings.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Handwriting predictions not found at {csv_path}\n"
                "Run src/unimodal/handwriting/train_handwriting_svm_embeddings.py first."
            )
        df = pd.read_csv(csv_path)
        prob_col = "oof_proba"
        label_col = "label"
    elif model == "best":
        csv_path = (
            OUTPUTS_ROOT
            / "unimodal_handwriting"
            / "final_model"
            / "all_models_oof_predictions.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Handwriting all-models predictions not found at {csv_path}\n"
                "Run src/unimodal/handwriting/benchmark_handwriting_models.py first."
            )
        df_all = pd.read_csv(csv_path)
        from sklearn.metrics import roc_auc_score

        best_model_name = None
        best_auc = -1.0
        for m_name, m_df in df_all.groupby("model"):
            subj = m_df.groupby("subject_id").agg(
                y_true=("label", "first"),
                y_proba=("oof_proba", "mean"),
            )
            try:
                auc = roc_auc_score(subj["y_true"], subj["y_proba"])
            except Exception:
                auc = 0.0
            if auc > best_auc:
                best_auc = auc
                best_model_name = m_name
        df = df_all[df_all["model"] == best_model_name].copy()
        prob_col = "oof_proba"
        label_col = "label"
    else:
        raise ValueError(f"Unknown handwriting model '{model}'. Choose 'svm' or 'best'.")

    agg = (
        df.groupby("subject_id")
        .agg(y_true=(label_col, "first"), p_pd=(prob_col, "mean"))
        .reset_index()
    )
    p_pd = agg["p_pd"].values.astype(float)
    y_proba = np.column_stack([1.0 - p_pd, p_pd])

    return {
        "subject_ids": agg["subject_id"].astype(str).values,
        "y_true": agg["y_true"].values.astype(int),
        "y_proba": y_proba,
        "task": "diagnosis",
        "note": (
            f"PaHaW + UCI handwriting: PD vs HC (model={model}, "
            f"drawing→subject aggregation by mean). N={len(agg)} subjects."
        ),
    }


def load_speech_predictions(speech_model: str = "catboost") -> dict:
    tsv_path = SPEECH_ROOT / "oof_probs.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Speech OOF probabilities not found at {tsv_path}\n"
            "Run the speech training notebook/script first."
        )
    df = pd.read_csv(tsv_path, sep="\t")

    if speech_model == "catboost":
        p_pd = df["p_catboost"].values
    elif speech_model == "cnn":
        p_pd = df["p_cnn"].values
    elif speech_model == "mean":
        p_pd = (df["p_catboost"].values + df["p_cnn"].values) / 2.0
    else:
        raise ValueError(
            f"Unknown speech_model '{speech_model}'. Choose 'catboost', 'cnn', or 'mean'."
        )

    p_pd = p_pd.astype(float)
    y_proba = np.column_stack([1.0 - p_pd, p_pd])

    return {
        "subject_ids": df["participant_id"].astype(str).values,
        "y_true": df["y"].values.astype(int),
        "y_proba": y_proba,
        "task": "diagnosis",
        "note": f"Speech ({speech_model}): PD vs HC classification. N={len(df)} participants.",
    }


def load_all(
    handwriting_model: str = "svm",
    speech_model: str = "catboost",
    include_gait: bool = True,
) -> dict[str, dict]:
    modalities = {}
    if include_gait:
        try:
            modalities["gait"] = load_gait_predictions()
        except FileNotFoundError as exc:
            print(f"[WARNING] Skipping gait — {exc}")
    try:
        modalities["handwriting"] = load_handwriting_predictions(model=handwriting_model)
    except FileNotFoundError as exc:
        print(f"[WARNING] Skipping handwriting — {exc}")
    try:
        modalities["speech"] = load_speech_predictions(speech_model=speech_model)
    except FileNotFoundError as exc:
        print(f"[WARNING] Skipping speech — {exc}")
    return modalities


def load_embedding_modalities(include_gait: bool = True) -> dict[str, dict]:
    modalities = {}
    if include_gait:
        try:
            modalities["gait"] = load_gait_concat_embeddings()
        except FileNotFoundError as exc:
            print(f"[WARNING] Skipping gait embeddings — {exc}")
    return modalities
