# BiomedAI: Multimodal Parkinson's Disease Replication Workspace

This directory is the structured project workspace for replication and extension of multimodal Parkinson's Disease prediction pipelines.

It is organized for practical experimentation across three modalities:

- Speech
- Gait
- Handwriting

and for staged development from EDA to unimodal, bimodal, trimodal, inference, and explainability workflows.

## 1. Project Objectives

- Reproduce baseline multimodal modeling behavior from the reference implementation.
- Build cleaner, reproducible pipelines for unimodal and multimodal training.
- Add clinically meaningful improvements in data alignment, cross-validation, fusion strategy, and explainability.
- Keep the workspace easy to navigate for team collaboration and semester milestones.

## 2. Repository Layout

Top-level structure in this project folder:

- `README.md`: This guide.
- `CONTRIBUTING.md`: Team workflow and contribution conventions.
- `requirements.txt`: Python dependencies for the replica workflows.
- `comprehensive_strategy_with_budget.md`: End-to-end execution plan and budget strategy.
- `literature/`: Background papers and reference material grouped by modality.
- `project_guidelines/`: Course/project instruction PDFs and supporting docs.
- `todo/`: Planning and feasibility notes for model improvements.
- `replica/`: Main implementation workspace.

### `replica/` substructure

- `01_eda/`: Exploratory data analysis notebooks and notes.
- `02_unimodal/`: Unimodal model training pipelines.
- `03_bimodal/`: Bimodal fusion experiments.
- `04_trimodal/`: Trimodal fusion experiments.
- `05_inference/`: Inference utilities and deployment-oriented scripts.
- `data/`: Local datasets and raw/intermediate assets (ignored by git).
- `outputs/`: Generated plots, models, metrics, reports (ignored by git).

## 3. Environment Setup

Recommended setup uses Conda with Python 3.10.

```bash
conda create -n biomedai python=3.10 -y
conda activate biomedai
pip install -r requirements.txt
```

If your environment does not use Conda, create an equivalent virtual environment and install from `requirements.txt`.

## 4. Data Organization

Expected data root:

- `replica/data/`

Typical modality-specific layout:

- `replica/data/gait/`
- `replica/data/speech/` or dataset-specific speech folders
- `replica/data/handwriting/`

Important:

- Large datasets and generated outputs are intentionally ignored in git.
- Keep data acquisition and preprocessing steps documented in modality-specific READMEs.

## 5. Common Entry Points

### EDA

- Primary notebooks in `replica/01_eda/`
- Example: `replica/01_eda/gait_eda.ipynb` or `replica/01_eda/weargait_eda.ipynb`

### Unimodal gait pipeline

- `replica/02_unimodal/gait/train_gait.py`
- `replica/02_unimodal/gait/train_gait_rf_replica.py`
- `replica/02_unimodal/gait/train_weargait_embeddings.py`
- `replica/02_unimodal/gait/prepare_weargait_index.py`

### SLURM batch scripts

- `replica/02_unimodal/gait/train_gait.slurm`
- `replica/02_unimodal/gait/train_weargait_embeddings.slurm`

## 6. Running Typical Workflows

### 6.1 Run EDA notebook

```bash
cd replica/01_eda
jupyter lab
```

### 6.2 Run unimodal gait training (local)

```bash
cd replica/02_unimodal/gait
python train_gait.py
```

### 6.3 Run unimodal gait training (cluster)

```bash
cd replica/02_unimodal/gait
sbatch train_gait.slurm
```

## 7. Outputs and Experiment Artifacts

Generated outputs should be written under:

- `replica/outputs/`

Examples:

- fold-level metrics
- confusion matrices
- saved model checkpoints
- summary JSON/CSV reports

Use consistent naming for reproducibility, for example:

- modality
- experiment tag
- fold or seed
- timestamp when needed

## 8. Naming and Organization Conventions

This workspace follows these conventions:

- Prefer lowercase filenames with underscores for long names.
- Keep implementation scripts in pipeline folders and analysis in notebooks.
- Keep planning docs in `todo/` and formal guidance in `project_guidelines/`.
- Avoid machine-specific absolute paths in committed code/docs.

## 9. Cleanup and Git Hygiene

Git is configured to ignore:

- dataset folders under `replica/data/`
- generated artifacts under `replica/outputs/`
- Python caches and notebook checkpoint files
- log and scheduler output files

Before sharing changes:

- verify no large datasets or generated artifacts are staged
- ensure docs reflect code/path changes
- ensure notebooks run at least through sanity cells

## 10. Team Collaboration

- Start from a feature branch.
- Keep pull requests focused and reviewable.
- Document assumptions and expected outputs for every new training/inference script.
- Prefer reproducible scripts over ad hoc notebook-only workflows for model training.

See `CONTRIBUTING.md` for contribution workflow details.

## 11. Current Focus Areas

- Stabilize unimodal pipelines and data indexing.
- Maintain subject-level splits and leakage-safe validation.
- Expand robust multimodal fusion and explainability analyses.
- Keep outputs interpretable and clinically meaningful.
