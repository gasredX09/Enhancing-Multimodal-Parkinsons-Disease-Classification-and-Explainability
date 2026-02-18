#!/usr/bin/env python3
"""
Project Setup Verification Script
Verify that all project files and folders are properly created
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/aryansharanreddyguda/biomedAI/project/replica")

# Expected directories
EXPECTED_DIRS = [
    "01_eda",
    "02_unimodal",
    "03_bimodal",
    "04_trimodal",
    "05_inference",
    "data",
    "outputs",
]

# Expected documentation files
EXPECTED_DOCS = [
    "00_SETUP_COMPLETE.md",
    "GETTING_STARTED.md",
    "QUICKSTART.md",
    "INDEX.md",
    "README.md",
    "STRUCTURE.md",
]

# Expected in 01_eda
EXPECTED_EDA_FILES = ["gait_eda.ipynb", "README.md"]


def check_directory_exists(path, name):
    """Check if a directory exists"""
    if path.is_dir():
        print(f"✅ {name}: {path}")
        return True
    else:
        print(f"❌ {name}: Missing - {path}")
        return False


def check_file_exists(path, name):
    """Check if a file exists"""
    if path.is_file():
        size_kb = path.stat().st_size / 1024
        print(f"✅ {name}: {path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"❌ {name}: Missing - {path}")
        return False


def main():
    print("=" * 80)
    print("PROJECT SETUP VERIFICATION")
    print("=" * 80)
    print()

    all_good = True

    # Check root directory exists
    if not PROJECT_ROOT.is_dir():
        print(f"❌ Project root not found: {PROJECT_ROOT}")
        sys.exit(1)

    print("🔍 Checking directories...")
    print("-" * 80)
    for dir_name in EXPECTED_DIRS:
        dir_path = PROJECT_ROOT / dir_name
        if not check_directory_exists(dir_path, dir_name):
            all_good = False
    print()

    print("🔍 Checking documentation files...")
    print("-" * 80)
    for doc_name in EXPECTED_DOCS:
        doc_path = PROJECT_ROOT / doc_name
        if not check_file_exists(doc_path, doc_name):
            all_good = False
    print()

    print("🔍 Checking EDA notebook...")
    print("-" * 80)
    eda_dir = PROJECT_ROOT / "01_eda"
    for file_name in EXPECTED_EDA_FILES:
        file_path = eda_dir / file_name
        if not check_file_exists(file_path, f"01_eda/{file_name}"):
            all_good = False
    print()

    print("=" * 80)
    if all_good:
        print("✅ PROJECT SETUP VERIFICATION PASSED!")
        print()
        print("Next steps:")
        print("  1. Read: GETTING_STARTED.md")
        print("  2. Set up virtual environment")
        print("  3. Run: jupyter lab 01_eda/gait_eda.ipynb")
        sys.exit(0)
    else:
        print("❌ SOME FILES ARE MISSING!")
        print("Please check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
