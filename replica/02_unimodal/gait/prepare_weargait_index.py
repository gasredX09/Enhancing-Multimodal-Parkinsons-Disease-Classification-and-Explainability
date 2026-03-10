from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


TASKS_DEFAULT = ["SelfPace", "HurriedPace", "TUG"]


def build_default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]  # project/replica
    data_root = repo_root / "data" / "gait" / "weargait"
    out_csv = repo_root / "outputs" / "unimodal_gait" / "weargait_index.csv"
    return data_root, out_csv


def infer_group(path_str: str) -> int:
    s = path_str.upper()
    if "PD PARTICIPANTS" in s:
        return 1
    return 0


def infer_subject(file_name: str) -> str:
    # Example: NLS002_SelfPace.csv -> NLS002
    return file_name.split("_", 1)[0].strip()


def infer_task(file_name: str) -> str:
    base = file_name.rsplit(".", 1)[0]
    return base.split("_", 1)[1] if "_" in base else "Unknown"


def collect_files(data_root: Path, allowed_tasks: List[str]) -> pd.DataFrame:
    rows = []
    allowed = set(allowed_tasks)

    for csv_path in sorted(data_root.rglob("*.csv")):
        path_str = str(csv_path)
        if "CSV files" not in path_str:
            continue

        fname = csv_path.name
        task = infer_task(fname)

        # Skip derived variants in v1. Keeps protocol compact and cleaner.
        if task.endswith("_mat") or task.endswith("_matTURN"):
            continue
        if task not in allowed:
            continue

        rows.append(
            {
                "subject_id": infer_subject(fname),
                "task": task,
                "label": infer_group(path_str),
                "file_path": str(csv_path),
            }
        )

    if not rows:
        raise ValueError(f"No WearGait CSV files matched task filters under: {data_root}")

    df = pd.DataFrame(rows).drop_duplicates()
    return df.sort_values(["subject_id", "task", "file_path"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    data_root, out_csv = build_default_paths()
    parser = argparse.ArgumentParser(description="Build WearGait file index for DL/embedding pipeline")
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--out-csv", type=Path, default=out_csv)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS_DEFAULT,
        help="Tasks to include (default: SelfPace HurriedPace TUG)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = collect_files(args.data_root, args.tasks)
    df.to_csv(args.out_csv, index=False)

    n_subjects = df["subject_id"].nunique()
    n_pd = df.loc[df["label"] == 1, "subject_id"].nunique()
    n_hc = df.loc[df["label"] == 0, "subject_id"].nunique()

    print(f"Saved index: {args.out_csv}")
    print(f"Rows: {len(df)} | Subjects: {n_subjects} | PD: {n_pd} | HC: {n_hc}")
    print("Task counts:")
    print(df["task"].value_counts().to_string())


if __name__ == "__main__":
    main()
