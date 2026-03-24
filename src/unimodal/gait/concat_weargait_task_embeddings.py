from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS_DEFAULT = ["SelfPace", "HurriedPace", "TUG"]


def build_default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    input_root = repo_root / "outputs" / "unimodal_gait" / "weargait_dl_embeddings"
    output_dir = repo_root / "outputs" / "unimodal_gait" / "weargait_concat_embeddings"
    return input_root, output_dir


def parse_args() -> argparse.Namespace:
    input_root, output_dir = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Concatenate subject-level WearGait embeddings across task-specific models."
    )
    parser.add_argument("--input-root", type=Path, default=input_root)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    parser.add_argument(
        "--missing-policy",
        choices=["intersection", "zero_fill"],
        default="intersection",
        help=(
            "How to handle subjects missing one or more task embeddings. "
            "'intersection' keeps only complete subjects; "
            "'zero_fill' keeps the union and fills missing tasks with zeros."
        ),
    )
    return parser.parse_args()


def load_task_embeddings(task_dir: Path) -> tuple[pd.DataFrame, int]:
    npz_path = task_dir / "weargait_subject_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    subject_ids = data["subject_ids"].astype(str)
    y = data["y"].astype(np.int64)
    x_emb = data["X_emb"].astype(np.float32)
    emb_dim = int(x_emb.shape[1])

    cols = [f"emb_{i:03d}" for i in range(emb_dim)]
    df = pd.DataFrame(x_emb, columns=cols)
    df.insert(0, "y", y)
    df.insert(0, "subject_id", subject_ids)
    return df, emb_dim


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_task: dict[str, pd.DataFrame] = {}
    emb_dims: dict[str, int] = {}

    for task in args.tasks:
        df, emb_dim = load_task_embeddings(args.input_root / task)
        rename_map = {c: f"{task}_{c}" for c in df.columns if c not in {"subject_id", "y"}}
        per_task[task] = df.rename(columns=rename_map)
        emb_dims[task] = emb_dim

    if args.missing_policy == "intersection":
        merged = None
        for task in args.tasks:
            df = per_task[task]
            if merged is None:
                merged = df.copy()
            else:
                merged = merged.merge(df, on=["subject_id", "y"], how="inner")
        assert merged is not None
    else:
        merged = None
        for task in args.tasks:
            df = per_task[task]
            if merged is None:
                merged = df.copy()
                continue

            merged = merged.merge(df, on="subject_id", how="outer", suffixes=("", f"_{task}"))
            if f"y_{task}" in merged.columns:
                merged["y"] = merged["y"].fillna(merged[f"y_{task}"])
                mismatch = (
                    merged["y"].notna()
                    & merged[f"y_{task}"].notna()
                    & (merged["y"] != merged[f"y_{task}"])
                )
                if bool(mismatch.any()):
                    bad = merged.loc[mismatch, "subject_id"].tolist()[:5]
                    raise ValueError(
                        f"Label mismatch while merging task '{task}' for subjects: {bad}"
                    )
                merged = merged.drop(columns=[f"y_{task}"])

        assert merged is not None
        feature_cols = [c for c in merged.columns if c not in {"subject_id", "y"}]
        merged.loc[:, feature_cols] = merged.loc[:, feature_cols].fillna(0.0)

    merged = merged.sort_values("subject_id").reset_index(drop=True)
    feature_cols = [c for c in merged.columns if c not in {"subject_id", "y"}]
    if not feature_cols:
        raise ValueError("No embedding columns found after concatenation.")

    out_npz = args.output_dir / "weargait_concat_subject_embeddings.npz"
    np.savez_compressed(
        out_npz,
        subject_ids=merged["subject_id"].astype(str).to_numpy(),
        y=merged["y"].astype(np.int64).to_numpy(),
        X_emb=merged.loc[:, feature_cols].to_numpy(dtype=np.float32),
        feature_names=np.array(feature_cols, dtype="U64"),
    )

    out_csv = args.output_dir / "weargait_concat_subject_embeddings.csv"
    merged.to_csv(out_csv, index=False)

    summary = {
        "tasks": args.tasks,
        "missing_policy": args.missing_policy,
        "n_subjects": int(len(merged)),
        "embedding_dims_by_task": emb_dims,
        "concatenated_embedding_dim": int(len(feature_cols)),
        "outputs": {
            "embeddings_npz": str(out_npz),
            "embeddings_csv": str(out_csv),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {out_npz}")
    print(f"  {out_csv}")
    print(f"  {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
