"""
Evaluate the multimodal late fusion model.

Run from the project root:
    python -m src.multimodal_fusion.evaluate

Saves fitted model to outputs/multimodal_fusion/fusion_model.json
and metrics to outputs/multimodal_fusion/metrics_table.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .loaders import load_all_from_embeddings
from .fusion import LateFusionModel, _apply_platt


def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    # Expected Calibration Error: weighted mean |predicted prob - observed freq| across bins.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_proba[mask].mean())
    return float(ece)


def _auc_bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    # Return (auc, ci_lo, ci_hi) via stratified bootstrap resampling.
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = y_true[idx], y_proba[idx]
        if yt.std() == 0:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except ValueError:
            pass
    arr = np.array(aucs)
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.percentile(arr, alpha * 100)), float(np.percentile(arr, (1 - alpha) * 100))


def _point_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    # Sensitivity, specificity, F1, accuracy, and Brier at threshold=0.5.
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "brier":       float(brier_score_loss(y_true, y_proba)),
    }


def compute_metrics_table(
    modalities: dict,
    model: LateFusionModel,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    AUC-ROC with 95% bootstrap CI, sensitivity, specificity, F1, accuracy, Brier
    for each modality on calibrated OOF predictions.
    """
    rows: list[dict] = []
    for name, d in modalities.items():
        y     = d["y_true"].astype(int)
        p_raw = d["y_proba"][:, 1]
        p_cal = (
            _apply_platt(p_raw, model.platt_params_[name])
            if name in model.platt_params_
            else p_raw
        )
        auc, ci_lo, ci_hi = _auc_bootstrap_ci(y, p_cal, n_bootstrap=n_bootstrap, seed=seed)
        pm = _point_metrics(y, p_cal)
        rows.append({
            "modality":    name,
            "auc":         round(auc,              4),
            "auc_ci_lo":   round(ci_lo,             4),
            "auc_ci_hi":   round(ci_hi,             4),
            "sensitivity": round(pm["sensitivity"], 4),
            "specificity": round(pm["specificity"], 4),
            "f1":          round(pm["f1"],          4),
            "accuracy":    round(pm["accuracy"],    4),
            "brier":       round(pm["brier"],       4),
            "note":        "calibrated OOF on real labels",
        })
    col_order = [
        "modality", "auc", "auc_ci_lo", "auc_ci_hi",
        "sensitivity", "specificity", "f1", "accuracy", "brier", "note",
    ]
    return pd.DataFrame(rows)[col_order]


def _banner(text: str) -> None:
    print(f"\n{'=' * 65}\n  {text}\n{'=' * 65}")


def main(strategy: str = "auc_weighted", save: bool = True) -> None:
    modalities = load_all_from_embeddings()
    if not modalities:
        print("[ERROR] No modalities loaded — check embedding files in src/multimodal_fusion/embeddings/")
        return

    _banner("Loaded modalities")
    for name, d in modalities.items():
        print(f"  {name:>12s}: N={len(d['y_true']):>4d}  prevalence={d['y_true'].mean():.2f}")

    model = LateFusionModel(strategy=strategy, calibrate=True)
    model.fit(modalities)
    _banner(f"Fitted LateFusionModel (strategy='{strategy}')")
    for name, w in model.weights_.items():
        print(f"  {name:>12s}: weight={w:.4f}  calibrated_auc={model.modality_aucs_[name]:.4f}")

    _banner("Unimodal metrics (95% bootstrap CI, 1000 iterations)")
    table = compute_metrics_table(modalities, model, n_bootstrap=1000)
    fmt_cols = ["auc", "auc_ci_lo", "auc_ci_hi", "sensitivity", "specificity", "f1", "accuracy", "brier"]
    header = (
        f"  {'Modality':<28s}  {'AUC':>6s}  {'CI Lo':>6s}  {'CI Hi':>6s}  "
        f"{'Sens':>6s}  {'Spec':>6s}  {'F1':>6s}  {'Acc':>6s}  {'Brier':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in table.iterrows():
        vals = "  ".join(
            f"{row[c]:>6.4f}" if not pd.isna(row[c]) else f"{'—':>6s}"
            for c in fmt_cols
        )
        print(f"  {row['modality']:<28s}  {vals}")

    if save:
        csv_path = (
            Path(__file__).parent.parent.parent
            / "outputs" / "multimodal_fusion" / "metrics_table.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")

    _banner("Calibration quality: raw vs Platt-scaled")
    print(f"  {'Modality':>12s}  {'':>8s}  {'Brier':>7s}  {'ECE':>7s}  {'LogLoss':>8s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}")
    for name, d in modalities.items():
        p_raw = d["y_proba"][:, 1]
        y     = d["y_true"]
        brier_raw = brier_score_loss(y, p_raw)
        ece_raw   = _ece(y, p_raw)
        ll_raw    = log_loss(y, p_raw)
        print(f"  {name:>12s}  {'raw':>8s}  {brier_raw:>7.4f}  {ece_raw:>7.4f}  {ll_raw:>8.4f}")

        if name in model.platt_params_:
            p_cal     = _apply_platt(p_raw, model.platt_params_[name])
            brier_cal = brier_score_loss(y, p_cal)
            ece_cal   = _ece(y, p_cal)
            ll_cal    = log_loss(y, p_cal)

            def _arrow(before: float, after: float) -> str:
                sign = "-" if after < before else "+"
                return f"({sign}{abs(after - before):.3f})"

            print(
                f"  {'':>12s}  {'Platt':>8s}  "
                f"{brier_cal:>7.4f}  {ece_cal:>7.4f}  {ll_cal:>8.4f}  "
                f"Brier {_arrow(brier_raw, brier_cal)}  "
                f"ECE {_arrow(ece_raw, ece_cal)}  "
                f"LL {_arrow(ll_raw, ll_cal)}"
            )
        print()

    if save:
        out_path = (
            Path(__file__).parent.parent.parent
            / "outputs" / "multimodal_fusion" / "fusion_model.json"
        )
        model.save(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate multimodal late fusion model.")
    parser.add_argument(
        "--strategy",
        default="auc_weighted",
        choices=["equal", "auc_weighted", "softmax_auc_weighted", "confidence_weighted"],
    )
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = parser.parse_args()
    main(strategy=args.strategy, save=args.save)
