from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EMBEDDING_PATHS = {
    "SelfPace": "weargait_dl_embeddings/SelfPace/weargait_subject_embeddings.npz",
    "HurriedPace": "weargait_dl_embeddings/HurriedPace/weargait_subject_embeddings.npz",
    "TUG": "weargait_dl_embeddings/TUG/weargait_subject_embeddings.npz",
    "All3": "weargait_concat_embeddings/weargait_concat_subject_embeddings.npz",
}


def build_defaults() -> tuple[Path, Path]:
    outputs_root = Path(__file__).resolve().parents[5] / "outputs" / "unimodal_gait"
    default_run_name = f"weargait_update3_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = outputs_root / "runs" / default_run_name
    return outputs_root, run_dir


def parse_args() -> argparse.Namespace:
    outputs_root, run_dir = build_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Run Update-3 gait-only ablations over task-specific WearGait embeddings "
            "with repeated CV, regularized models, and dimensionality reduction."
        )
    )
    parser.add_argument("--outputs-root", type=Path, default=outputs_root)
    parser.add_argument("--run-dir", type=Path, default=run_dir)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_fresh_run_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def load_embedding(npz_path: Path) -> dict[str, np.ndarray]:
    arr = np.load(npz_path, allow_pickle=True)
    return {
        "subject_ids": arr["subject_ids"].astype(str),
        "y": arr["y"].astype(int),
        "X": arr["X_emb"].astype(np.float32),
    }


def load_all_embeddings(outputs_root: Path) -> dict[str, dict[str, np.ndarray]]:
    loaded = {}
    for name, rel_path in EMBEDDING_PATHS.items():
        loaded[name] = load_embedding(outputs_root / rel_path)
    return loaded


def align_to_subjects(item: dict[str, np.ndarray], subject_ids: list[str]) -> dict[str, np.ndarray]:
    idx_map = {sid: i for i, sid in enumerate(item["subject_ids"].tolist())}
    idx = [idx_map[sid] for sid in subject_ids]
    return {
        "subject_ids": np.array(subject_ids, dtype=str),
        "y": item["y"][idx],
        "X": item["X"][idx],
    }


def build_representations(
    loaded: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    common_all = sorted(
        set(loaded["SelfPace"]["subject_ids"].tolist())
        & set(loaded["HurriedPace"]["subject_ids"].tolist())
        & set(loaded["TUG"]["subject_ids"].tolist())
    )
    common_sp_tug = sorted(
        set(loaded["SelfPace"]["subject_ids"].tolist())
        & set(loaded["TUG"]["subject_ids"].tolist())
    )

    selfpace = align_to_subjects(loaded["SelfPace"], common_all)
    tug = align_to_subjects(loaded["TUG"], common_all)
    hurried = align_to_subjects(loaded["HurriedPace"], common_all)
    all3 = align_to_subjects(loaded["All3"], common_all)

    selfpace_tug_sp = align_to_subjects(loaded["SelfPace"], common_sp_tug)
    selfpace_tug_tug = align_to_subjects(loaded["TUG"], common_sp_tug)

    representations = {
        "SelfPace": selfpace,
        "TUG": tug,
        "SelfPace+TUG": {
            "subject_ids": selfpace_tug_sp["subject_ids"],
            "y": selfpace_tug_sp["y"],
            "X": np.concatenate([selfpace_tug_sp["X"], selfpace_tug_tug["X"]], axis=1),
        },
        "All3": all3,
        "HurriedPace": hurried,
    }
    subject_counts = {
        "common_all3_subjects": len(common_all),
        "common_selfpace_tug_subjects": len(common_sp_tug),
    }
    return representations, subject_counts


def get_model_configs(seed: int) -> dict[str, Pipeline]:
    return {
        "lr_l2": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l2")),
            ]
        ),
        "lr_l1": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l1")),
            ]
        ),
        "pca64_lr_l2": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=64, random_state=seed)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l2")),
            ]
        ),
        "pca32_lr_l2": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=32, random_state=seed)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l2")),
            ]
        ),
        "mi128_lr_l2": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=mutual_info_classif, k=128)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l2")),
            ]
        ),
    }


def clamp_model_to_dim(model_name: str, base_model: Pipeline, n_features: int) -> tuple[str, Pipeline]:
    if model_name == "pca64_lr_l2" and n_features < 64:
        model_name = f"pca{n_features}_lr_l2"
        base_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_features)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l2")),
            ]
        )
    elif model_name == "pca32_lr_l2" and n_features < 32:
        model_name = f"pca{n_features}_lr_l2"
        base_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_features)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l2")),
            ]
        )
    elif model_name == "mi128_lr_l2" and n_features < 128:
        model_name = f"mi{n_features}_lr_l2"
        base_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=mutual_info_classif, k=n_features)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", penalty="l2")),
            ]
        )
    return model_name, base_model


def evaluate_representation(
    representation_name: str,
    item: dict[str, np.ndarray],
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    X = item["X"]
    y = item["y"]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    split_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    for model_name, model in get_model_configs(seed).items():
        effective_name, effective_model = clamp_model_to_dim(model_name, model, X.shape[1])
        model_split_rows = []

        for split_id, (tr, va) in enumerate(cv.split(X, y), start=1):
            effective_model.fit(X[tr], y[tr])
            proba = effective_model.predict_proba(X[va])[:, 1]
            pred = (proba >= 0.5).astype(int)
            model_split_rows.append(
                {
                    "representation": representation_name,
                    "model": effective_name,
                    "split": split_id,
                    "accuracy": accuracy_score(y[va], pred),
                    "f1": f1_score(y[va], pred, zero_division=0),
                    "auc": roc_auc_score(y[va], proba) if len(np.unique(y[va])) > 1 else np.nan,
                    "ap": average_precision_score(y[va], proba) if len(np.unique(y[va])) > 1 else np.nan,
                }
            )

        model_split_df = pd.DataFrame(model_split_rows)
        split_rows.extend(model_split_rows)
        summary_rows.append(
            {
                "representation": representation_name,
                "model": effective_name,
                "subjects": len(y),
                "embedding_dim": int(X.shape[1]),
                "mean_accuracy": float(model_split_df["accuracy"].mean()),
                "mean_f1": float(model_split_df["f1"].mean()),
                "mean_auc": float(model_split_df["auc"].mean()),
                "mean_ap": float(model_split_df["ap"].mean()),
                "ci95_accuracy_lo": float(model_split_df["accuracy"].quantile(0.025)),
                "ci95_accuracy_hi": float(model_split_df["accuracy"].quantile(0.975)),
                "ci95_f1_lo": float(model_split_df["f1"].quantile(0.025)),
                "ci95_f1_hi": float(model_split_df["f1"].quantile(0.975)),
                "ci95_auc_lo": float(model_split_df["auc"].quantile(0.025)),
                "ci95_auc_hi": float(model_split_df["auc"].quantile(0.975)),
                "ci95_ap_lo": float(model_split_df["ap"].quantile(0.025)),
                "ci95_ap_hi": float(model_split_df["ap"].quantile(0.975)),
            }
        )

    return pd.DataFrame(split_rows), summary_rows


def make_auc_heatmap(summary_df: pd.DataFrame, out_path: Path) -> None:
    pivot = summary_df.pivot(index="representation", columns="model", values="mean_auc")
    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(11, 6))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.5,
        vmax=0.9,
        cbar_kws={"label": "Mean AUC"},
    )
    ax.set_title("WearGait Update-3 Ablation: Representation x Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_top_models_plot(summary_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary_df.sort_values("mean_auc", ascending=False).head(8).copy()
    plot_df["label"] = plot_df["representation"] + " | " + plot_df["model"]
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x="mean_auc", y="label", palette="crest")
    ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("Mean AUC")
    ax.set_ylabel("")
    ax.set_title("Top Update-3 Gait Configurations")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(
            f"{width:.3f}",
            (width, patch.get_y() + patch.get_height() / 2),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(5, 0),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_notes(
    run_dir: Path,
    summary_df: pd.DataFrame,
    subject_counts: dict[str, int],
    args: argparse.Namespace,
) -> None:
    best = summary_df.iloc[0]
    best_per_repr = (
        summary_df.sort_values(["representation", "mean_auc"], ascending=[True, False])
        .groupby("representation")
        .head(1)
    )
    text = f"""# WearGait Update-3 Gait-Only Ablation Notes

Run directory: `{run_dir}`

## What this run does
- Compares task-level gait representations: `TUG`, `SelfPace`, `SelfPace+TUG`, `All3`, and `HurriedPace`
- Evaluates regularized downstream classifiers and dimensionality-reduction variants
- Uses repeated stratified CV for more stable gait-only reporting

## Subject coverage
- Common all-task cohort: `{subject_counts['common_all3_subjects']}`
- Common SelfPace+TUG cohort: `{subject_counts['common_selfpace_tug_subjects']}`

## Best overall configuration
- Representation: `{best['representation']}`
- Model: `{best['model']}`
- Mean AUC: `{best['mean_auc']:.3f}`
- Mean F1: `{best['mean_f1']:.3f}`
- 95% AUC interval: `{best['ci95_auc_lo']:.3f}` to `{best['ci95_auc_hi']:.3f}`

## Best model per representation
{best_per_repr[['representation', 'model', 'mean_auc', 'mean_f1', 'mean_accuracy']].to_string(index=False)}

## Interpretation
- This run directly addresses the remaining gait-only Update-3 questions: task ablation, regularized downstream evaluation, and whether reduced-dimensional models help.
- The recommended gait branch for the next internal baseline should be the strongest representation-model pair from this table.

## Run settings
- n_splits: `{args.n_splits}`
- n_repeats: `{args.n_repeats}`
- seed: `{args.seed}`
"""
    (run_dir / "notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_fresh_run_dir(args.run_dir)

    loaded = load_all_embeddings(args.outputs_root)
    representations, subject_counts = build_representations(loaded)

    split_frames = []
    summary_rows = []
    for representation_name in ["TUG", "SelfPace", "SelfPace+TUG", "All3", "HurriedPace"]:
        split_df, rows = evaluate_representation(
            representation_name=representation_name,
            item=representations[representation_name],
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            seed=args.seed,
        )
        split_frames.append(split_df)
        summary_rows.extend(rows)

    per_split_df = pd.concat(split_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)

    per_split_path = args.run_dir / "per_split_metrics.csv"
    summary_path = args.run_dir / "summary_metrics.csv"
    heatmap_path = args.run_dir / "representation_model_auc_heatmap.png"
    top_plot_path = args.run_dir / "top_configurations_auc.png"

    per_split_df.to_csv(per_split_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    make_auc_heatmap(summary_df, heatmap_path)
    make_top_models_plot(summary_df, top_plot_path)
    write_notes(args.run_dir, summary_df, subject_counts, args)

    best_per_repr = (
        summary_df.sort_values(["representation", "mean_auc"], ascending=[True, False])
        .groupby("representation")
        .head(1)
        .to_dict(orient="records")
    )
    run_summary = {
        "run_dir": str(args.run_dir),
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "subject_counts": subject_counts,
        "best_overall": summary_df.iloc[0].to_dict(),
        "best_per_representation": best_per_repr,
        "outputs": {
            "per_split_metrics_csv": str(per_split_path),
            "summary_metrics_csv": str(summary_path),
            "representation_model_auc_heatmap_png": str(heatmap_path),
            "top_configurations_auc_png": str(top_plot_path),
            "notes_md": str(args.run_dir / "notes.md"),
        },
    }
    (args.run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(json.dumps(run_summary, indent=2))
    print("\nSummary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
