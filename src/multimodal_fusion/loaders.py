"""
Standardized loaders for each unimodal model's OOF predictions.

Each loader returns a dict with keys:
    subject_ids : np.ndarray of str, shape (N,)
    y_true      : np.ndarray of int, shape (N,)   — ground truth labels
    y_proba     : np.ndarray of float, shape (N, 2) — [P(class0), P(class1)]
    task        : str   — 'diagnosis' (PD vs HC) or 'severity' (mild vs mod/severe PD)
    note        : str   — human-readable description

All loaders aggregate to subject level before returning.

NOTE on subject overlap: gait, handwriting, and speech use entirely different
datasets (PDFE, PaHaW/UCI, PC-GITA or similar). There is no subject overlap
across modalities, so fusion weights are derived from per-modality AUC
rather than from a joint held-out set.

NOTE on task mismatch: The PDFE gait model classifies PD *severity*
(mild=0 vs moderate/severe=1) on PD-only subjects. Handwriting and speech
classify PD vs HC (diagnosis). Interpret gait probability carefully — it
is not directly comparable to the other two, but is included as a signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# Project paths (resolved relative to this file)
_SRC = Path(__file__).parent.parent          # src/
_PROJECT = _SRC.parent                        # project/
OUTPUTS_ROOT = _PROJECT / "outputs"
SPEECH_ROOT = _SRC / "unimodal" / "speech" / "scripts"


def load_gait_predictions() -> dict:
    """
    Load PDFE TCN gait predictions (subject-level, already aggregated).

    Task: PD severity — mild (0) vs moderate/severe (1), PD-only subjects.
    Dataset: Figshare PDFE, N=35 PD patients.
    """
    npz_path = OUTPUTS_ROOT / "unimodal_gait" / "PDFE_Severity_Classification" / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Gait predictions not found at {npz_path}\n"
            "Run src/unimodal/gait/train_gait.py first."
        )
    data = np.load(npz_path, allow_pickle=True)
    return {
        "subject_ids": data["subject_ids"].astype(str),
        "y_true":      data["y_true"].astype(int),
        "y_proba":     data["y_proba"].astype(float),   # (N, 2)
        "task":        "severity",
        "note":        (
            "PDFE Figshare: PD-only severity classification "
            "(mild=0 vs moderate/severe=1). N=35."
        ),
    }


def load_handwriting_predictions(model: str = "svm") -> dict:
    """
    Load handwriting OOF predictions, aggregated from drawing level to subject level.

    model: 'svm' uses the SVM+PCA pipeline (default).
           'best' uses the best model from all_models_oof_predictions.csv
                  (selects the model with highest mean OOF AUC: gpc_rbf).

    Task: PD vs HC diagnosis.
    Dataset: PaHaW + UCI handwriting, N~108 drawings / ~72 subjects.
    """
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
        # Pick the model with the highest mean OOF AUC per subject
        from sklearn.metrics import roc_auc_score
        best_model_name = None
        best_auc = -1.0
        for m_name, m_df in df_all.groupby("model"):
            # Aggregate to subject level first, then score
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

    # Aggregate drawing-level probabilities to subject level
    agg = (
        df.groupby("subject_id")
        .agg(
            y_true=(label_col, "first"),
            p_pd=(prob_col, "mean"),
        )
        .reset_index()
    )
    p_pd = agg["p_pd"].values.astype(float)
    y_proba = np.column_stack([1.0 - p_pd, p_pd])

    return {
        "subject_ids": agg["subject_id"].astype(str).values,
        "y_true":      agg["y_true"].values.astype(int),
        "y_proba":     y_proba,
        "task":        "diagnosis",
        "note":        (
            f"PaHaW + UCI handwriting: PD vs HC (model={model}, "
            f"drawing→subject aggregation by mean). "
            f"N={len(agg)} subjects."
        ),
    }


def load_speech_predictions(speech_model: str = "catboost") -> dict:
    """
    Load speech OOF predictions (subject-level).

    speech_model:
        'catboost' — CatBoost on static acoustic features (AUC ~0.826, best single model)
        'cnn'      — CNN on mel spectrograms (AUC ~0.775)
        'mean'     — average of catboost and cnn probabilities

    Task: PD vs HC diagnosis.
    Dataset: PC-GITA or similar, N=290 participants.
    """
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
        "y_true":      df["y"].values.astype(int),
        "y_proba":     y_proba,
        "task":        "diagnosis",
        "note":        (
            f"Speech ({speech_model}): PD vs HC classification. "
            f"N={len(df)} participants."
        ),
    }


def load_all(
    handwriting_model: str = "svm",
    speech_model: str = "catboost",
    include_gait: bool = True,
) -> dict[str, dict]:
    """
    Convenience loader: returns all available modalities as a dict.

    Keys: 'gait', 'handwriting', 'speech'

    Args:
        handwriting_model: 'svm' or 'best'  (see load_handwriting_predictions)
        speech_model:      'catboost', 'cnn', or 'mean'  (see load_speech_predictions)
        include_gait:      If False, skips gait (useful when task mismatch matters).
    """
    modalities = {}
    if include_gait:
        try:
            modalities["gait"] = load_gait_predictions()
        except FileNotFoundError as e:
            print(f"[WARNING] Skipping gait — {e}")
    try:
        modalities["handwriting"] = load_handwriting_predictions(model=handwriting_model)
    except FileNotFoundError as e:
        print(f"[WARNING] Skipping handwriting — {e}")
    try:
        modalities["speech"] = load_speech_predictions(speech_model=speech_model)
    except FileNotFoundError as e:
        print(f"[WARNING] Skipping speech — {e}")
    return modalities
