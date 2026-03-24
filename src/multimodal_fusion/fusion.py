"""
Late fusion model for multimodal Parkinson's Disease classification.

Since gait, handwriting, and speech predictions come from entirely disjoint
subject sets, we cannot train a stacking meta-learner on a joint held-out set.
Instead, fusion weights are derived from each modality's own held-out AUC.

Supported strategies
--------------------
equal
    All modalities contribute equally (w_i = 1 / n_modalities).

auc_weighted
    Weights are proportional to each modality's ROC AUC on its own held-out data.
    Better-performing modalities get more weight.

softmax_auc_weighted
    Same as auc_weighted but weights are passed through a softmax, which makes
    the weight distribution smoother and avoids a single modality dominating.

confidence_weighted  (inference-time only)
    Weights are proportional to model confidence = 1 - prediction entropy.
    High-entropy (uncertain) predictions are down-weighted per sample.
    Requires per-sample probabilities; can be mixed with auc_weighted.

Inference
---------
At inference time, supply a dict of per-modality P(PD) values for one patient:

    model = LateFusionModel(strategy="auc_weighted")
    model.fit(modality_data)   # modality_data from loaders.load_all()

    # New patient: only speech and handwriting available
    p_fused = model.predict_proba({"handwriting": 0.82, "speech": 0.61})

If a modality is missing for a patient, the remaining modalities are
re-normalized to sum to 1.0.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

STRATEGIES = Literal["equal", "auc_weighted", "softmax_auc_weighted", "confidence_weighted"]


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of a Bernoulli(p) distribution, in nats."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _compute_metrics(y_true: np.ndarray, y_proba_1: np.ndarray) -> dict:
    """Return a standard metrics dict for a binary classifier."""
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

    Parameters
    ----------
    strategy : str
        One of 'equal', 'auc_weighted', 'softmax_auc_weighted',
        'confidence_weighted'.
    """

    def __init__(self, strategy: str = "auc_weighted") -> None:
        if strategy not in ("equal", "auc_weighted", "softmax_auc_weighted", "confidence_weighted"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: "
                "'equal', 'auc_weighted', 'softmax_auc_weighted', 'confidence_weighted'."
            )
        self.strategy = strategy
        # Set after fit()
        self.modality_names_: list[str] = []
        self.modality_aucs_: dict[str, float] = {}
        self.weights_: dict[str, float] = {}
        self.modality_metrics_: dict[str, dict] = {}
        self.modality_notes_: dict[str, str] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, modality_data: dict[str, dict]) -> "LateFusionModel":
        """
        Compute per-modality metrics and derive fusion weights.

        Parameters
        ----------
        modality_data : dict
            Mapping of modality name → loader dict (from loaders.load_all()).
            Each value must have keys: 'y_true', 'y_proba', 'note'.

        Returns
        -------
        self
        """
        self.modality_names_ = list(modality_data.keys())
        if len(self.modality_names_) == 0:
            raise ValueError("No modalities provided.")

        for name, data in modality_data.items():
            y_true = np.asarray(data["y_true"])
            y_proba = np.asarray(data["y_proba"])  # (N, 2)
            p1 = y_proba[:, 1]

            metrics = _compute_metrics(y_true, p1)
            self.modality_metrics_[name] = metrics
            self.modality_aucs_[name] = metrics["auc"]
            self.modality_notes_[name] = data.get("note", "")

        self._set_weights()
        self._fitted = True
        return self

    def _set_weights(self) -> None:
        """Compute static fusion weights from per-modality AUCs."""
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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        modality_proba: dict[str, float | np.ndarray],
    ) -> float:
        """
        Fuse per-modality probabilities for one or more patients.

        Parameters
        ----------
        modality_proba : dict
            Mapping of modality name → P(PD) scalar or array.
            Missing modalities are handled gracefully: remaining weights
            are re-normalized to sum to 1.

        Returns
        -------
        float or np.ndarray
            Fused P(PD).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        present = {k: np.asarray(v, dtype=float) for k, v in modality_proba.items()
                   if k in self.weights_}
        if not present:
            raise ValueError("None of the provided modality names match fitted modalities.")

        unknown = set(modality_proba) - set(self.weights_)
        if unknown:
            warnings.warn(f"Ignoring unknown modalities: {unknown}")

        # Re-normalize weights over present modalities
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
        """Return binary class prediction (0=HC, 1=PD)."""
        p = self.predict_proba(modality_proba)
        return (np.asarray(p) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_unimodal(self) -> dict[str, dict]:
        """Return per-modality metrics computed during fit()."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.modality_metrics_

    def evaluate_all_strategies(
        self, modality_data: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Evaluate all four fusion strategies on a simulated co-modal test set.

        Because subjects are disjoint across modalities, we cannot compute a
        true fused AUC. Instead, this method:
          1. Reports each modality's own held-out AUC (ground truth).
          2. Simulates fusion performance via bootstrap pairing: randomly
             sample one subject from each modality, fuse their probabilities,
             and evaluate against random paired labels over many iterations.

        This gives a rough sense of expected fusion behaviour but is NOT a
        substitute for a real multi-modal test cohort.

        Returns
        -------
        dict mapping strategy name → metrics dict.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        results: dict[str, dict] = {}

        # Per-modality results (no fusion)
        for name, data in modality_data.items():
            y_true = np.asarray(data["y_true"])
            y_proba = np.asarray(data["y_proba"])
            results[f"unimodal_{name}"] = _compute_metrics(y_true, y_proba[:, 1])
            results[f"unimodal_{name}"]["note"] = data.get("note", "")

        # Bootstrap simulation of fused AUC
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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the fitted model to a JSON-compatible dict."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return {
            "strategy":         self.strategy,
            "modality_names":   self.modality_names_,
            "weights":          self.weights_,
            "modality_aucs":    self.modality_aucs_,
            "modality_metrics": self.modality_metrics_,
            "modality_notes":   self.modality_notes_,
        }

    def save(self, path: str | Path) -> None:
        """Save the fitted model configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Fusion model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LateFusionModel":
        """Restore a previously saved fusion model from JSON."""
        with open(path) as f:
            d = json.load(f)
        model = cls(strategy=d["strategy"])
        model.modality_names_   = d["modality_names"]
        model.weights_          = d["weights"]
        model.modality_aucs_    = d["modality_aucs"]
        model.modality_metrics_ = d["modality_metrics"]
        model.modality_notes_   = d.get("modality_notes", {})
        model._fitted = True
        return model

    def __repr__(self) -> str:
        if not self._fitted:
            return f"LateFusionModel(strategy='{self.strategy}', fitted=False)"
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in self.weights_.items())
        return f"LateFusionModel(strategy='{self.strategy}', weights=[{w_str}])"
