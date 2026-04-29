# BiomedAI Project Workspace

This directory is the actively maintained workspace for multimodal Parkinson's disease modeling and explainability work.

It includes:

- Notebook-based EDA for gait, WearGait, handwriting, and speech assets.
- Unimodal training pipelines for gait and speech.
- Collected datasets and experiment outputs.
- Project planning and setup documentation.

The older replica folder layout has been flattened into a cleaner top-level structure.

## Repository Layout

```text
project/
├── data/
│   ├── gait/
│   └── handwriting/
├── docs/
│   ├── guidelines/
│   ├── planning/
│   ├── roadmap/
│   └── setup/
├── notebooks/
│   └── eda/
├── outputs/
│   ├── gait_eda/
│   ├── unimodal_gait/
│   └── unimodal_gait_rf/
├── scripts/
│   └── verify_setup.py
├── src/
│   └── unimodal/
│       ├── gait/
│       └── speech/
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Environment

```bash
conda create -n biomedai python=3.10 -y
conda activate biomedai
pip install -r requirements.txt
```

### 2. Verify Folder Structure

```bash
python scripts/verify_setup.py
```

### 3. Run EDA Notebooks

```bash
cd notebooks/eda
jupyter lab
```

Recommended starting notebook:

- gait_eda.ipynb

### 4. Run Unimodal Gait Pipelines

From the project root:

```bash
python src/unimodal/gait/train_gait.py
python src/unimodal/gait/prepare_weargait_index.py
python src/unimodal/gait/train_weargait_embeddings.py
python src/unimodal/gait/train_gait_rf.py
```

## Data Organization

- Input datasets are under data/ and grouped by modality.
- Generated artifacts are written to outputs/.
- Large raw data and generated files are ignored for future additions via .gitignore.

Notes:

- Some historical data and outputs are already tracked in Git from earlier snapshots.
- Keep new large files out of commits unless explicitly needed for reproducibility.

## Source Code Map

### Gait

- src/unimodal/gait/train_gait.py
	- TCN-based supervised severity classification from figshare IMU gait files.
- src/unimodal/gait/prepare_weargait_index.py
	- Builds WearGait manifest CSV for downstream modeling.
- src/unimodal/gait/train_weargait_embeddings.py
	- Trains WearGait model and exports embeddings.
- src/unimodal/gait/train_gait_rf.py
	- Random forest baseline replication using engineered features.

### Speech

- src/unimodal/speech/scripts/
	- Static-feature sorting, QC, and benchmark scripts.
	- Includes notebook and generated benchmark artifacts from prior runs.

## Documentation Map

- docs/guidelines/: course and platform guidance PDFs
- docs/planning/: strategy and planning docs
- docs/roadmap/: improvement assessments and workflow notes
- docs/setup/: migrated setup and legacy replica documentation

## SLURM / Cluster Runs

Gait SLURM launchers are in:

- src/unimodal/gait/train_gait.slurm
- src/unimodal/gait/train_weargait_embeddings.slurm

These scripts assume the project path root at:

- /ocean/projects/med260006p/shared/biomedAI/project

Adjust environment/module activation as needed for your cluster account.

## Contributing

Use the workflow in CONTRIBUTING.md.

Short version:

- work on feature branches
- keep changes focused
- update docs when behavior changes
- avoid committing large data dumps and transient artifacts

