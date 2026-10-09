from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the winning Update-3 gait embedding transform (TUG + StandardScaler + PCA32)."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("project/outputs/unimodal_gait/weargait_dl_embeddings/TUG/weargait_subject_embeddings.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh output directory for the exported transformed embedding.",
    )
    parser.add_argument("--n-components", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    data = np.load(args.input_npz, allow_pickle=True)
    subject_ids = data["subject_ids"].astype(str)
    y = data["y"].astype(int)
    X = data["X_emb"].astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=args.n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    feature_names = np.array([f"tug_pca32_{i:02d}" for i in range(X_pca.shape[1])], dtype=str)
    out_npz = args.output_dir / "tug_pca32_subject_embeddings.npz"
    np.savez_compressed(
        out_npz,
        subject_ids=subject_ids,
        y=y,
        X_emb=X_pca,
        feature_names=feature_names,
    )

    metadata = {
        "source_embedding": str(args.input_npz),
        "output_embedding": str(out_npz),
        "n_subjects": int(len(subject_ids)),
        "input_dim": int(X.shape[1]),
        "output_dim": int(X_pca.shape[1]),
        "transform": {
            "scaler": "StandardScaler",
            "pca": {
                "n_components": int(args.n_components),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            },
        },
        "note": (
            "Full-cohort transformed export of the winning Update-3 representation: "
            "TUG embedding followed by StandardScaler and PCA(32)."
        ),
    }
    (args.output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
