from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export additional best Update-3 gait embedding artifacts: All3+MI128 and TUG PCA32 CSV."
    )
    parser.add_argument(
        "--all3-input-npz",
        type=Path,
        default=Path("project/outputs/unimodal_gait/weargait_concat_embeddings/weargait_concat_subject_embeddings.npz"),
    )
    parser.add_argument(
        "--tug-input-npz",
        type=Path,
        default=Path("project/outputs/unimodal_gait/weargait_dl_embeddings/TUG/weargait_subject_embeddings.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Existing output directory where additional export artifacts will be written.",
    )
    parser.add_argument("--tug-pca-components", type=int, default=32)
    parser.add_argument("--all3-mi-k", type=int, default=128)
    return parser.parse_args()


def export_tug_pca32_csv(tug_npz: Path, output_dir: Path, n_components: int) -> dict:
    data = np.load(tug_npz, allow_pickle=True)
    subject_ids = data["subject_ids"].astype(str)
    y = data["y"].astype(int)
    X = data["X_emb"].astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    cols = [f"tug_pca32_{i:02d}" for i in range(X_pca.shape[1])]
    df = pd.DataFrame(X_pca, columns=cols)
    df.insert(0, "y", y)
    df.insert(0, "subject_id", subject_ids)
    out_csv = output_dir / "tug_pca32_subject_embeddings.csv"
    df.to_csv(out_csv, index=False)

    return {
        "path": str(out_csv),
        "n_subjects": int(len(subject_ids)),
        "output_dim": int(X_pca.shape[1]),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def export_all3_mi128(all3_npz: Path, output_dir: Path, k: int) -> dict:
    data = np.load(all3_npz, allow_pickle=True)
    subject_ids = data["subject_ids"].astype(str)
    y = data["y"].astype(int)
    X = data["X_emb"].astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_scaled.shape[1]))
    X_sel = selector.fit_transform(X_scaled, y).astype(np.float32)

    support = selector.get_support(indices=True)
    if "feature_names" in data:
        source_feature_names = data["feature_names"].astype(str)
        selected_names = source_feature_names[support]
    else:
        selected_names = np.array([f"all3_mi128_{i:03d}" for i in range(X_sel.shape[1])], dtype=str)

    out_npz = output_dir / "all3_mi128_subject_embeddings.npz"
    np.savez_compressed(
        out_npz,
        subject_ids=subject_ids,
        y=y,
        X_emb=X_sel,
        feature_names=selected_names,
    )

    return {
        "path": str(out_npz),
        "n_subjects": int(len(subject_ids)),
        "input_dim": int(X.shape[1]),
        "output_dim": int(X_sel.shape[1]),
        "feature_names_preview": selected_names[:10].tolist(),
    }


def main() -> None:
    args = parse_args()
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {args.output_dir}")

    tug_csv = export_tug_pca32_csv(args.tug_input_npz, args.output_dir, args.tug_pca_components)
    all3_npz = export_all3_mi128(args.all3_input_npz, args.output_dir, args.all3_mi_k)

    metadata = {
        "tug_pca32_csv": tug_csv,
        "all3_mi128_npz": all3_npz,
        "note": (
            "Additional full-cohort transformed exports for Update-3: "
            "TUG PCA32 as CSV and All3 with MI-based feature selection to 128 dimensions."
        ),
    }
    (args.output_dir / "additional_export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
