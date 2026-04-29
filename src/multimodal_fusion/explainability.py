"""
SHAP-based explainability for the multimodal late fusion model.

FusionExplainer: modality-level SHAP contributions to fused P(PD).
GaitEmbeddingExplainer: feature-level SHAP within the gait embedding space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _apply_platt(p: np.ndarray, params: dict) -> np.ndarray:
    eps = 1e-7
    return expit(params["a"] * logit(np.clip(p, eps, 1 - eps)) + params["b"])


def _apply_prior(p: np.ndarray, prev_source: float, prev_target: float = 0.5) -> np.ndarray:
    eps = 1e-7
    shift = logit(np.clip(prev_target, eps, 1 - eps)) - logit(np.clip(prev_source, eps, 1 - eps))
    return expit(logit(np.clip(p, eps, 1 - eps)) + shift)


def _calibrate(model, name: str, p: np.ndarray) -> np.ndarray:
    """Apply the same calibration steps the fusion model uses at inference."""
    if model.calibrate and name in model.platt_params_:
        p = _apply_platt(p, model.platt_params_[name])
    if model.normalize_prevalence and name in model.modality_prevalences_:
        p = _apply_prior(p, model.modality_prevalences_[name])
    return p


class FusionExplainer:
    """
    Exact SHAP decomposition for AUC-weighted late fusion.
    SHAP_i = w_i * (p_i_cal - E[p_i_cal]); positive = pushed toward PD.
    Missing modalities have their weights renormalized across present ones.
    """

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, model, modality_data: dict[str, dict]) -> "FusionExplainer":
        # Compute per-modality baseline expectations from training data.
        self.model_ = model
        self.expected_cal_: dict[str, float] = {}

        for name, d in modality_data.items():
            p = np.asarray(d["y_proba"])[:, 1]
            p_cal = _calibrate(model, name, p)
            self.expected_cal_[name] = float(p_cal.mean())

        weights = model.weights_
        self.baseline_: float = float(
            sum(weights[n] * self.expected_cal_[n] for n in weights)
        )
        self._fitted = True
        return self

    def explain(self, modality_proba: dict[str, float]) -> dict[str, float]:
        # Return per-modality SHAP contributions for one patient (raw P(PD) as input).
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        model = self.model_
        present = {k: v for k, v in modality_proba.items() if k in model.weights_}
        if not present:
            raise ValueError("No recognised modalities in modality_proba.")

        raw_w = np.array([model.weights_[k] for k in present])
        raw_w = raw_w / raw_w.sum()
        norm_weights = {k: float(raw_w[i]) for i, k in enumerate(present)}

        result: dict[str, float] = {"baseline": self.baseline_}
        prediction = self.baseline_

        for name, p_raw in present.items():
            p_cal = float(_calibrate(model, name, np.array([p_raw]))[0])
            contrib = norm_weights[name] * (p_cal - self.expected_cal_[name])
            result[name] = round(float(contrib), 6)
            prediction += contrib

        result["prediction"] = round(float(prediction), 6)
        return result

    def explain_batch(self, records: list[dict[str, float]]) -> pd.DataFrame:
        # Explain a list of patients; returns DataFrame with baseline, modality SHAPs, prediction.
        rows = [self.explain(r) for r in records]
        return pd.DataFrame(rows).fillna(0.0)

    def summary(self, modality_data: dict[str, dict]) -> pd.DataFrame:
        # Mean |SHAP|, mean signed SHAP, std, and weight per modality across training subjects.
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        records = []
        for name, d in modality_data.items():
            p1 = np.asarray(d["y_proba"])[:, 1]
            records.extend({name: float(p)} for p in p1)

        df = self.explain_batch(records)
        modality_cols = [c for c in df.columns if c not in ("baseline", "prediction")]
        rows = []
        for col in modality_cols:
            rows.append({
                "modality":      col,
                "mean_abs_shap": float(df[col].abs().mean()),
                "mean_shap":     float(df[col].mean()),
                "std_shap":      float(df[col].std()),
                "weight":        float(self.model_.weights_.get(col, 0.0)),
            })
        return pd.DataFrame(rows).set_index("modality").sort_values(
            "mean_abs_shap", ascending=False
        )


class GaitEmbeddingExplainer:
    """
    Feature-level SHAP for gait embeddings via SHAP LinearExplainer on L2 logistic regression.
    Auto-detects PCA format (tug_pca32_XX) vs task format (SelfPace_*, HurriedPace_*, TUG_*).
    Trained on all subjects — not the same OOF model used in fusion, but decision boundary is near-identical.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._is_pca_format = False

    def fit(self, gait_emb_dict: dict) -> "GaitEmbeddingExplainer":
        # Train logistic regression on gait embeddings and fit SHAP LinearExplainer.
        import shap

        x = gait_emb_dict["X_emb"].astype(float)
        y = gait_emb_dict["y_true"].astype(int)
        feature_names = gait_emb_dict.get("feature_names")

        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x)

        self.clf_ = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        self.clf_.fit(x_scaled, y)

        self.feature_names_ = (
            np.array(feature_names) if feature_names is not None
            else np.array([f"feat_{i}" for i in range(x.shape[1])])
        )
        self.subject_ids_ = gait_emb_dict.get("subject_ids")
        self.y_true_ = y

        self._is_pca_format = any("pca" in f.lower() for f in self.feature_names_)

        # LinearExplainer is exact for logistic regression
        self.explainer_ = shap.LinearExplainer(self.clf_, x_scaled)
        self.shap_values_ = self.explainer_.shap_values(x_scaled)  # (N, D)
        self._fitted = True
        return self

    def explain(self, x_emb: np.ndarray) -> np.ndarray:
        # SHAP values for one or more new subjects (raw unscaled embeddings as input).
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        x = np.atleast_2d(x_emb).astype(float)
        x_scaled = self.scaler_.transform(x)
        sv = self.explainer_.shap_values(x_scaled)
        return sv.squeeze() if x_emb.ndim == 1 else sv

    def pc_importances(self, subjects: str = "all", n_groups: int = 4) -> dict[str, float]:
        # Mean |SHAP| per PC quartile group. subjects: 'all', 'pd', or 'hc'.
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        mask = (
            self.y_true_ == 1 if subjects == "pd"
            else self.y_true_ == 0 if subjects == "hc"
            else np.ones(len(self.y_true_), dtype=bool)
        )
        sv = np.abs(self.shap_values_[mask])
        n_features = sv.shape[1]
        group_size = max(1, n_features // n_groups)

        result = {}
        for g in range(n_groups):
            lo = g * group_size
            hi = min((g + 1) * group_size, n_features)
            label = f"PC{lo+1}-{hi}"
            result[label] = float(sv[:, lo:hi].mean())
        return result

    def task_importances(self, subjects: str = "all") -> dict[str, float]:
        # Mean |SHAP| per walking task (SelfPace/HurriedPace/TUG). Raises if PCA format.
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._is_pca_format:
            raise ValueError(
                "Current gait embeddings are PCA-format — use pc_importances() instead."
            )

        mask = (
            self.y_true_ == 1 if subjects == "pd"
            else self.y_true_ == 0 if subjects == "hc"
            else np.ones(len(self.y_true_), dtype=bool)
        )
        sv = np.abs(self.shap_values_[mask])
        tasks: dict[str, float] = {}
        for task in ("SelfPace", "HurriedPace", "TUG"):
            feat_mask = np.array([f.startswith(task) for f in self.feature_names_])
            if feat_mask.any():
                tasks[task] = float(sv[:, feat_mask].mean())
        return tasks

    def feature_importance_df(self, top_n: int = 20) -> pd.DataFrame:
        # Top-n features by mean |SHAP|; columns: feature, group, mean_abs_shap, mean_shap.
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        mean_shap = self.shap_values_.mean(axis=0)

        def _group(name: str) -> str:
            if self._is_pca_format:
                try:
                    idx = int(name.split("_")[-1])
                    n = len(self.feature_names_)
                    q = (idx * 4) // n
                    labels = ["Q1 (high variance)", "Q2", "Q3", "Q4 (low variance)"]
                    return labels[min(q, 3)]
                except (ValueError, IndexError):
                    return "unknown"
            for t in ("SelfPace", "HurriedPace", "TUG"):
                if name.startswith(t):
                    return t
            return "unknown"

        df = pd.DataFrame({
            "feature":       self.feature_names_,
            "group":         [_group(f) for f in self.feature_names_],
            "mean_abs_shap": mean_abs,
            "mean_shap":     mean_shap,
        })
        return (
            df.sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def subject_shap_df(self) -> pd.DataFrame:
        # (N_subjects × N_features) DataFrame of SHAP values with subject_id and y_true prepended.
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        df = pd.DataFrame(self.shap_values_, columns=self.feature_names_)
        df.insert(0, "y_true", self.y_true_)
        if self.subject_ids_ is not None:
            df.insert(0, "subject_id", self.subject_ids_)
        return df
