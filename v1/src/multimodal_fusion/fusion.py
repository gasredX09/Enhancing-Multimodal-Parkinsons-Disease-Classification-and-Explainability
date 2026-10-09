"""
Late fusion for multimodal PD classification.

Modalities are trained on disjoint cohorts, so fusion weights come from
each modality's own held-out AUC rather than a joint meta-learner.
Missing modalities at inference time are handled by renormalizing remaining weights.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import expit, logit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

STRATEGIES = Literal["equal", "auc_weighted", "softmax_auc_weighted", "confidence_weighted"]

_EPS = 1e-7


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    # Shannon entropy of a Bernoulli(p) distribution, in nats.
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _fit_platt(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    # Fit Platt scaling; returns {a, b} s.t. p_cal = sigmoid(a * logit(p) + b).
    x = logit(np.clip(p, _EPS, 1 - _EPS)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def _apply_platt(p: np.ndarray, params: dict[str, float]) -> np.ndarray:
    # Apply a fitted Platt scaler to probability array p.
    return expit(params["a"] * logit(np.clip(p, _EPS, 1 - _EPS)) + params["b"])


def _adjust_prior(
    p: np.ndarray, prev_source: float, prev_target: float
) -> np.ndarray:
    """
    Correct a probability calibrated to *prev_source* to instead reflect
    *prev_target* (prior / prevalence correction on the log-odds scale).

    log_odds_new = log_odds_raw - logit(prev_source) + logit(prev_target)
    """
    if abs(prev_source - prev_target) < 1e-6:
        return p
    shift = logit(np.clip(prev_target, _EPS, 1 - _EPS)) - logit(
        np.clip(prev_source, _EPS, 1 - _EPS)
    )
    return expit(logit(np.clip(p, _EPS, 1 - _EPS)) + shift)


def _compute_metrics(y_true: np.ndarray, y_proba_1: np.ndarray) -> dict:
    # Return a standard metrics dict for a binary classifier.
    y_pred = (y_proba_1 >= 0.5).astype(int)
    metrics: dict = {}
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba_1))
    except Exception:
        metrics["auc"] = float("nan")
    try:
        metrics["ap"] = float(average_precision_score(y_true, y_proba_1))
    except Exception:
        metrics["ap"] = float("nan")
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["n_subjects"] = int(len(y_true))
    metrics["prevalence"] = float(y_true.mean())
    return metrics


class LateFusionModel:
    """
    Late fusion of unimodal PD classifiers with disjoint training sets.
    Weights are derived from each modality's calibrated OOF AUC.
    """

    def __init__(
        self,
        strategy: str = "auc_weighted",
        calibrate: bool = True,
        normalize_prevalence: bool = False,
    ) -> None:
        if strategy not in ("equal", "auc_weighted", "softmax_auc_weighted", "confidence_weighted"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: "
                "'equal', 'auc_weighted', 'softmax_auc_weighted', 'confidence_weighted'."
            )
        self.strategy = strategy
        self.calibrate = calibrate
        self.normalize_prevalence = normalize_prevalence
        # Set after fit()
        self.modality_names_: list[str] = []
        self.modality_aucs_: dict[str, float] = {}
        self.weights_: dict[str, float] = {}
        self.modality_metrics_: dict[str, dict] = {}
        self.modality_notes_: dict[str, str] = {}
        self.platt_params_: dict[str, dict[str, float]] = {}  # name -> {a, b}
        self.modality_prevalences_: dict[str, float] = {}
        self._fitted = False

    # Fitting
    def fit(self, modality_data: dict[str, dict]) -> "LateFusionModel":
        """Compute per-modality metrics and derive fusion weights."""
        self.modality_names_ = list(modality_data.keys())
        if len(self.modality_names_) == 0:
            raise ValueError("No modalities provided.")

        for name, data in modality_data.items():
            y_true = np.asarray(data["y_true"])
            y_proba = np.asarray(data["y_proba"])  # (N, 2)
            p1 = y_proba[:, 1]

            self.modality_prevalences_[name] = float(y_true.mean())

            if self.calibrate:
                self.platt_params_[name] = _fit_platt(p1, y_true)
                p1 = _apply_platt(p1, self.platt_params_[name])

            metrics = _compute_metrics(y_true, p1)
            self.modality_metrics_[name] = metrics
            self.modality_aucs_[name] = metrics["auc"]
            self.modality_notes_[name] = data.get("note", "")

        self._set_weights()
        self._fitted = True
        return self

    def _set_weights(self) -> None:
        # Compute static fusion weights from per-modality AUCs.
        names = self.modality_names_
        aucs = np.array([self.modality_aucs_[n] for n in names], dtype=float)

        if self.strategy == "equal":
            raw = np.ones(len(names))
        elif self.strategy == "auc_weighted":
            raw = np.clip(aucs, 0.0, 1.0)
        elif self.strategy in ("softmax_auc_weighted", "confidence_weighted"):
            # For confidence_weighted, static weights start as softmax(AUC);
            # per-sample confidence adjustment happens in predict_proba.
            raw = softmax(aucs * 5.0)   # temperature=5 sharpens the distribution
        else:
            raw = np.ones(len(names))

        total = raw.sum()
        if total == 0:
            raw = np.ones(len(names))
            total = float(len(names))
        w = raw / total
        self.weights_ = {n: float(w[i]) for i, n in enumerate(names)}

    # Inference
    def predict_proba(
        self,
        modality_proba: dict[str, float | np.ndarray],
    ) -> float:
        # Fuse per-modality P(PD) values; missing modalities are dropped and weights renormalized.
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        present = {k: np.asarray(v, dtype=float) for k, v in modality_proba.items()
                   if k in self.weights_}
        if not present:
            raise ValueError("None of the provided modality names match fitted modalities.")

        unknown = set(modality_proba) - set(self.weights_)
        if unknown:
            warnings.warn(f"Ignoring unknown modalities: {unknown}")

        if self.calibrate and self.platt_params_:
            present = {
                k: _apply_platt(v, self.platt_params_[k])
                for k, v in present.items()
                if k in self.platt_params_
            }

        if self.normalize_prevalence and self.modality_prevalences_:
            present = {
                k: _adjust_prior(v, self.modality_prevalences_[k], 0.5)
                for k, v in present.items()
            }

        raw_w = np.array([self.weights_[k] for k in present], dtype=float)

        if self.strategy == "confidence_weighted":
            # Per-sample adjustment: multiply static weight by confidence = 1 - entropy
            probs = np.array(list(present.values()), dtype=float)  # (n_mod,) or (n_mod, N)
            entropies = _binary_entropy(probs)                       # same shape
            confidence = 1.0 - entropies / np.log(2)                # normalise to [0,1]
            # raw_w is (n_mod,); broadcast over samples if probs is 2-D
            if probs.ndim == 2:
                adj_w = raw_w[:, None] * np.clip(confidence, 0, 1)
                adj_w = adj_w / adj_w.sum(axis=0, keepdims=True)
            else:
                adj_w = raw_w * np.clip(confidence, 0, 1)
                adj_w = adj_w / adj_w.sum()
        else:
            raw_w = raw_w / raw_w.sum()
            adj_w = raw_w

        probs = np.array(list(present.values()), dtype=float)
        if probs.ndim == 2:
            # adj_w shape: (n_mod, N) for confidence_weighted, else (n_mod,)
            if adj_w.ndim == 1:
                fused = (probs * adj_w[:, None]).sum(axis=0)
            else:
                fused = (probs * adj_w).sum(axis=0)
        else:
            fused = float((probs * adj_w).sum())
        return fused

    def predict(self, modality_proba: dict[str, float | np.ndarray], threshold: float = 0.5):
        # Return binary class prediction (0=HC, 1=PD).
        p = self.predict_proba(modality_proba)
        return (np.asarray(p) >= threshold).astype(int)

    # Evaluation
    def evaluate_unimodal(self) -> dict[str, dict]:
        """Return per-modality metrics computed during fit()."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.modality_metrics_

    def evaluate_all_strategies(
        self, modality_data: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Evaluate all fusion strategies via bootstrap simulation.
        Subjects are disjoint so fused AUC is estimated by randomly pairing
        subjects across modalities — not a substitute for a real multimodal test set.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        results: dict[str, dict] = {}

        for name, data in modality_data.items():
            y_true = np.asarray(data["y_true"])
            y_proba = np.asarray(data["y_proba"])
            results[f"unimodal_{name}"] = _compute_metrics(y_true, y_proba[:, 1])
            results[f"unimodal_{name}"]["note"] = data.get("note", "")

        rng = np.random.default_rng(42)
        n_bootstrap = 2000
        n_subjects_per_iter = 50  # synthetic cohort size per bootstrap draw

        for strat in ("equal", "auc_weighted", "softmax_auc_weighted", "confidence_weighted"):
            orig_strategy = self.strategy
            self.strategy = strat
            self._set_weights()

            bootstrap_aucs: list[float] = []
            arrays = {
                name: (
                    np.asarray(data["y_true"]),
                    np.asarray(data["y_proba"])[:, 1],
                )
                for name, data in modality_data.items()
            }

            for _ in range(n_bootstrap):
                # Sample independently from each modality
                sampled_proba: dict[str, np.ndarray] = {}
                sampled_labels: dict[str, np.ndarray] = {}
                for name, (yt, yp) in arrays.items():
                    idx = rng.integers(0, len(yt), size=n_subjects_per_iter)
                    sampled_proba[name] = yp[idx]
                    sampled_labels[name] = yt[idx]

                # Fuse probabilities
                fused_p = self.predict_proba(sampled_proba)

                # Use majority-voted label across modalities as synthetic ground truth
                all_labels = np.stack(list(sampled_labels.values()), axis=0)  # (M, N)
                y_sim = (all_labels.mean(axis=0) >= 0.5).astype(int)

                if y_sim.std() == 0:
                    continue  # skip degenerate draws
                try:
                    bootstrap_aucs.append(roc_auc_score(y_sim, fused_p))
                except Exception:
                    pass

            self.strategy = orig_strategy
            self._set_weights()

            if bootstrap_aucs:
                arr = np.array(bootstrap_aucs)
                results[f"fusion_{strat}"] = {
                    "auc_mean":   float(arr.mean()),
                    "auc_std":    float(arr.std()),
                    "auc_ci_lo":  float(np.percentile(arr, 2.5)),
                    "auc_ci_hi":  float(np.percentile(arr, 97.5)),
                    "n_bootstrap": n_bootstrap,
                    "warning": (
                        "Bootstrap-simulated AUC on synthetic co-modal cohort. "
                        "Labels are majority-voted across modalities. "
                        "NOT a substitute for a real multi-modal test set."
                    ),
                }

        return results

    # Serialisation
    def to_dict(self) -> dict:
        """Serialise the fitted model to a JSON-compatible dict."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return {
            "strategy":              self.strategy,
            "calibrate":             self.calibrate,
            "normalize_prevalence":  self.normalize_prevalence,
            "modality_names":        self.modality_names_,
            "weights":               self.weights_,
            "modality_aucs":         self.modality_aucs_,
            "modality_metrics":      self.modality_metrics_,
            "modality_notes":        self.modality_notes_,
            "platt_params":          self.platt_params_,
            "modality_prevalences":  self.modality_prevalences_,
        }

    def save(self, path: str | Path) -> None:
        """Save the fitted model configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Fusion model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LateFusionModel":
        # Restore a previously saved fusion model from JSON.
        with open(path) as f:
            d = json.load(f)
        model = cls(
            strategy=d["strategy"],
            calibrate=d.get("calibrate", False),
            normalize_prevalence=d.get("normalize_prevalence", False),
        )
        model.modality_names_        = d["modality_names"]
        model.weights_               = d["weights"]
        model.modality_aucs_         = d["modality_aucs"]
        model.modality_metrics_      = d["modality_metrics"]
        model.modality_notes_        = d.get("modality_notes", {})
        model.platt_params_          = d.get("platt_params", {})
        model.modality_prevalences_  = d.get("modality_prevalences", {})
        model._fitted = True
        return model

    def __repr__(self) -> str:
        if not self._fitted:
            return (
                f"LateFusionModel(strategy='{self.strategy}', "
                f"calibrate={self.calibrate}, fitted=False)"
            )
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in self.weights_.items())
        flags = []
        if self.calibrate:
            flags.append("calibrated")
        if self.normalize_prevalence:
            flags.append("prev_norm")
        flag_str = (", " + "+".join(flags)) if flags else ""
        return f"LateFusionModel(strategy='{self.strategy}'{flag_str}, weights=[{w_str}])"


# Stacking meta-learner
class StackingFusionModel:
    """
    Stacking meta-learner trained on bootstrap-simulated synthetic triplets.

    Since cohorts are disjoint there is no real joint training set, so subjects
    are randomly paired across modalities and labeled by majority vote.
    Weights are learned rather than fixed to AUC, but all metrics are on
    synthetic data — not equivalent to a real multimodal evaluation.
    """

    def __init__(
        self,
        meta_clf=None,
        calibrate: bool = True,
        n_synthetic: int = 100,
        n_bootstrap: int = 500,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.meta_clf      = meta_clf
        self.calibrate     = calibrate
        self.n_synthetic   = n_synthetic
        self.n_bootstrap   = n_bootstrap
        self.cv_folds      = cv_folds
        self.random_state  = random_state

        self.modality_names_:  list[str] = []
        self.platt_params_:    dict[str, dict] = {}
        self.meta_clf_:        object = None   # fitted on full synthetic dataset
        self.coef_:            np.ndarray | None = None
        self.cv_aucs_:         np.ndarray | None = None
        self._fitted = False

    def _default_clf(self):
        return LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)

    def _calibrated_proba(
        self, modality_data: dict[str, dict]
    ) -> dict[str, np.ndarray]:
        """Return per-modality calibrated P(PD) arrays (1-D, N subjects)."""
        out = {}
        for name, d in modality_data.items():
            p = np.asarray(d["y_proba"])[:, 1]
            if self.calibrate and name in self.platt_params_:
                p = _apply_platt(p, self.platt_params_[name])
            out[name] = p
        return out

    def _build_synthetic_dataset(
        self,
        arrays: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample n synthetic subjects by drawing independently from each modality pool."""
        X_cols, label_cols = [], []
        for name in self.modality_names_:
            yt, yp = arrays[name]
            idx = rng.integers(0, len(yt), size=n)
            X_cols.append(yp[idx])
            label_cols.append(yt[idx])
        X = np.column_stack(X_cols)
        y = (np.stack(label_cols, axis=0).mean(axis=0) >= 0.5).astype(int)
        return X, y

    def fit(self, modality_data: dict[str, dict]) -> "StackingFusionModel":
        """Fit Platt scalers and meta-learner on bootstrap-simulated synthetic data."""
        from sklearn.model_selection import StratifiedKFold

        self.modality_names_ = list(modality_data.keys())
        clf = self.meta_clf or self._default_clf()

        if self.calibrate:
            for name, d in modality_data.items():
                p = np.asarray(d["y_proba"])[:, 1]
                y = np.asarray(d["y_true"])
                self.platt_params_[name] = _fit_platt(p, y)

        cal_arrays = self._calibrated_proba(modality_data)
        arrays = {
            name: (np.asarray(modality_data[name]["y_true"]), cal_arrays[name])
            for name in self.modality_names_
        }

        rng = np.random.default_rng(self.random_state)

        X_all, y_all = self._build_synthetic_dataset(
            arrays, rng, n=self.n_synthetic * self.cv_folds
        )

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True,
            random_state=self.random_state
        )
        cv_aucs: list[float] = []
        for train_idx, val_idx in skf.split(X_all, y_all):
            fold_clf = type(clf)(**clf.get_params())
            fold_clf.fit(X_all[train_idx], y_all[train_idx])
            if y_all[val_idx].std() == 0:
                continue
            try:
                p_val = fold_clf.predict_proba(X_all[val_idx])[:, 1]
                cv_aucs.append(roc_auc_score(y_all[val_idx], p_val))
            except Exception:
                pass
        self.cv_aucs_ = np.array(cv_aucs)

        X_full, y_full = self._build_synthetic_dataset(
            arrays, rng, n=self.n_synthetic * self.n_bootstrap
        )
        final_clf = type(clf)(**clf.get_params())
        final_clf.fit(X_full, y_full)
        self.meta_clf_  = final_clf
        self.coef_      = (
            np.asarray(final_clf.coef_[0])
            if hasattr(final_clf, "coef_") else None
        )
        self._fitted = True
        return self

    def predict_proba(
        self, modality_proba: dict[str, float | np.ndarray]
    ) -> float | np.ndarray:
        """Predict fused P(PD); missing modalities are filled with 0.5 before the meta-learner."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        is_scalar = all(np.isscalar(v) for v in modality_proba.values())
        n = 1 if is_scalar else len(next(iter(modality_proba.values())))

        cols = []
        for name in self.modality_names_:
            if name in modality_proba:
                p = np.asarray(modality_proba[name], dtype=float)
                if self.calibrate and name in self.platt_params_:
                    p = _apply_platt(p, self.platt_params_[name])
            else:
                p = np.full(n, 0.5)
            cols.append(p if not is_scalar else np.atleast_1d(p))

        X = np.column_stack(cols)
        p_fused = self.meta_clf_.predict_proba(X)[:, 1]
        return float(p_fused[0]) if is_scalar else p_fused

    def predict(self, modality_proba: dict[str, float | np.ndarray], threshold: float = 0.5) -> int | np.ndarray:
        """Return binary prediction (1=PD, 0=HC)."""
        p = self.predict_proba(modality_proba)
        return int(p >= threshold) if np.isscalar(p) else (p >= threshold).astype(int)

    def evaluate(self, modality_data: dict[str, dict], n_bootstrap: int = 1000) -> dict:
        """
        Evaluate the stacking model on bootstrap-simulated synthetic cohorts.

        Returns a dict with mean AUC, std, 95% CI, and per-modality learned weights.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        cal_arrays = self._calibrated_proba(modality_data)
        arrays = {
            name: (np.asarray(modality_data[name]["y_true"]), cal_arrays[name])
            for name in self.modality_names_
        }
        rng = np.random.default_rng(self.random_state + 1)
        aucs: list[float] = []

        for _ in range(n_bootstrap):
            X, y = self._build_synthetic_dataset(arrays, rng, n=self.n_synthetic)
            if y.std() == 0:
                continue
            try:
                p = self.meta_clf_.predict_proba(X)[:, 1]
                aucs.append(roc_auc_score(y, p))
            except Exception:
                pass

        arr = np.array(aucs)
        result = {
            "auc_mean":  float(arr.mean()),
            "auc_std":   float(arr.std()),
            "auc_ci_lo": float(np.percentile(arr, 2.5)),
            "auc_ci_hi": float(np.percentile(arr, 97.5)),
            "cv_auc_mean": float(self.cv_aucs_.mean()) if self.cv_aucs_ is not None else float("nan"),
            "cv_auc_std":  float(self.cv_aucs_.std())  if self.cv_aucs_ is not None else float("nan"),
            "warning": (
                "All metrics computed on bootstrap-simulated synthetic cohorts. "
                "Labels are majority-voted across disjoint modality datasets. "
                "NOT equivalent to a real multi-modal held-out evaluation."
            ),
        }
        if self.coef_ is not None:
            result["meta_learner_weights"] = {
                name: float(w)
                for name, w in zip(self.modality_names_, self.coef_)
            }
        return result

    def __repr__(self) -> str:
        if not self._fitted:
            return f"StackingFusionModel(calibrate={self.calibrate}, fitted=False)"
        cv_str = (
            f"cv_auc={self.cv_aucs_.mean():.3f}" if self.cv_aucs_ is not None else ""
        )
        w_str = (
            ", ".join(f"{n}={w:.3f}" for n, w in zip(self.modality_names_, self.coef_))
            if self.coef_ is not None else "n/a"
        )
        return f"StackingFusionModel({cv_str}, meta_weights=[{w_str}])"
