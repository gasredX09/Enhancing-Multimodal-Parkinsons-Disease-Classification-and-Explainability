# Enhancing Multimodal Parkinson's Disease Classification and Explainability

CMU PBAI capstone project. Multimodal PD classification fusing gait, handwriting, and speech via late fusion with SHAP-based explainability.

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
│       ├── gait_eda.ipynb
│       ├── handwriting_eda.ipynb
│       └── weargait_eda.ipynb
├── outputs/
│   ├── gait_eda/
│   ├── multimodal_fusion/
│   ├── unimodal_gait/
│   └── unimodal_gait_rf/
├── src/
│   ├── multimodal_fusion/
│   │   ├── embeddings/          # gait/speech .npz and handwriting .csv
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── explainability.py
│   │   ├── fusion.py
│   │   └── loaders.py
│   └── unimodal/
│       ├── gait/
│       │   ├── slurm/
│       │   ├── audit_weargait_coverage.py
│       │   ├── concat_weargait_task_embeddings.py
│       │   ├── ensemble_fusion.py
│       │   ├── gait_ensemble_orchestrator.py
│       │   ├── prepare_weargait_index.py
│       │   ├── train_gait.py
│       │   ├── train_gait_rf.py
│       │   └── train_weargait_embeddings.py
│       ├── handwriting/
│       │   ├── slurm/
│       │   ├── benchmark_handwriting_models.py
│       │   ├── finalize_handwriting_model.py
│       │   └── train_handwriting_svm_embeddings.py
│       └── speech/
│           └── scripts/
│               ├── TrainSpeechBasedModel_v2.0.ipynb
│               ├── findStaticOutliers.py
│               ├── sort_by_diagnosis.py
│               ├── sort_by_task.py
│               ├── sort_static.py
│               └── tryTrainForStaticFeatures.py
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

### 2. Run Multimodal Fusion Evaluation

From the project root:

```bash
python -m src.multimodal_fusion.evaluate
```

Outputs saved to `outputs/multimodal_fusion/`:
- `fusion_model.json` — fitted model weights and Platt parameters
- `metrics_table.csv` — per-modality AUC, sensitivity, specificity, F1, Brier

### 3. Run Unimodal Gait Pipeline

```bash
python src/unimodal/gait/prepare_weargait_index.py
python src/unimodal/gait/gait_ensemble_orchestrator.py --tasks weargait
python src/unimodal/gait/concat_weargait_task_embeddings.py
python src/unimodal/gait/train_gait.py
```

## Source Code Map

### Multimodal Fusion (`src/multimodal_fusion/`)

- `fusion.py` — `LateFusionModel`: AUC-weighted probability averaging with Platt calibration and prevalence normalization. `StackingFusionModel`: learned meta-classifier.
- `loaders.py` — loads gait/handwriting/speech embeddings from `embeddings/` and runs OOF CV to produce calibrated probability estimates.
- `evaluate.py` — end-to-end evaluation: fits the fusion model, computes metrics, saves outputs.
- `explainability.py` — `FusionExplainer`: modality-level SHAP contributions. `GaitEmbeddingExplainer`: feature-level SHAP within gait embedding space.

### Gait (`src/unimodal/gait/`)

- `train_gait.py` — TCN-based supervised classification from figshare IMU gait files.
- `train_weargait_embeddings.py` — trains task-specific WearGait models and exports per-task embeddings.
- `concat_weargait_task_embeddings.py` — concatenates SelfPace, HurriedPace, and TUG embeddings per subject.
- `gait_ensemble_orchestrator.py` — orchestrates the full WearGait embedding pipeline.
- `prepare_weargait_index.py` — builds WearGait manifest CSV for downstream modeling.
- `train_gait_rf.py` — random forest baseline using engineered features.
- `ensemble_fusion.py` — gait-internal ensemble fusion (weighted, stacking, voting).

### Handwriting (`src/unimodal/handwriting/`)

- `train_handwriting_svm_embeddings.py` — trains SVM and exports drawing-level embeddings.
- `finalize_handwriting_model.py` — selects and saves the final handwriting model.
- `benchmark_handwriting_models.py` — cross-validates multiple classifiers on handwriting features.

### Speech (`src/unimodal/speech/scripts/`)

- `TrainSpeechBasedModel_v2.0.ipynb` — CNN + CatBoost speech model training; exports `.npz` embeddings.
- Utility scripts for feature sorting, QC, and outlier detection.

## Data Organization

- Input datasets are under `data/` grouped by modality.
- Generated artifacts are written to `outputs/`.
- Embedding files (`.npz`, `.csv`) live in `src/multimodal_fusion/embeddings/` and are tracked in Git.
- Other large binary files (`.npy`, model weights) are gitignored.

## SLURM / Cluster Runs

SLURM launchers are in `src/unimodal/gait/slurm/` and `src/unimodal/handwriting/slurm/`. These assume the project root at:

```
/ocean/projects/med260006p/shared/biomedAI/project
```

## Contributing

See `CONTRIBUTING.md`. Short version: work on feature branches, keep changes focused, avoid committing large data dumps.
