#!/usr/bin/env python3
"""Verify the reorganized project structure."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "data",
    "docs",
    "docs/setup",
    "notebooks",
    "notebooks/eda",
    "outputs",
    "scripts",
    "src",
    "src/unimodal",
    "src/unimodal/gait",
    "src/unimodal/speech",
]

EXPECTED_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "requirements.txt",
    "notebooks/eda/gait_eda.ipynb",
    "src/unimodal/gait/train_gait.py",
    "src/unimodal/gait/train_gait_rf.py",
    "src/unimodal/gait/train_weargait_embeddings.py",
]


def check_directory_exists(path: Path, name: str) -> bool:
    if path.is_dir():
        print(f"OK DIR   {name}: {path}")
        return True
    print(f"MISSING  {name}: {path}")
    return False


def check_file_exists(path: Path, name: str) -> bool:
    if path.is_file():
        size_kb = path.stat().st_size / 1024
        print(f"OK FILE  {name}: {path} ({size_kb:.1f} KB)")
        return True
    print(f"MISSING  {name}: {path}")
    return False


def main() -> None:
    print("=" * 72)
    print("REORGANIZED PROJECT STRUCTURE CHECK")
    print("=" * 72)

    all_good = True

    if not PROJECT_ROOT.is_dir():
        print(f"MISSING PROJECT ROOT: {PROJECT_ROOT}")
        sys.exit(1)

    print("\nChecking directories\n" + "-" * 72)
    for rel in EXPECTED_DIRS:
        if not check_directory_exists(PROJECT_ROOT / rel, rel):
            all_good = False

    print("\nChecking files\n" + "-" * 72)
    for rel in EXPECTED_FILES:
        if not check_file_exists(PROJECT_ROOT / rel, rel):
            all_good = False

    old_replica = PROJECT_ROOT / "replica"
    print("\nChecking legacy directory\n" + "-" * 72)
    if old_replica.exists():
        print(f"UNEXPECTED legacy directory still exists: {old_replica}")
        all_good = False
    else:
        print("OK replica directory removed")

    print("\n" + "=" * 72)
    if all_good:
        print("PASS: structure is tidy and consistent.")
        print("Next: run a notebook and one training script to validate runtime paths.")
        sys.exit(0)

    print("FAIL: one or more required paths are missing.")
    sys.exit(1)


if __name__ == "__main__":
    main()
