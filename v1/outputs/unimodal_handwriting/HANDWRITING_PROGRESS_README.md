# Handwriting Progress README

Last updated: 2026-03-22  
Owner: Copilot-assisted implementation  
Scope: Unimodal handwriting pipeline (data merge + SVM embeddings + multi-model benchmark)

## 1) Objective

Build a reproducible handwriting modeling pipeline for Parkinson classification that:
- Merges UCI + PaHaW handwriting trajectories
- Produces per-drawing summary features
- Generates embedding outputs with SVM and 5-fold CV
- Benchmarks multiple models on the same folds
- Selects the best model by a primary metric (default ROC-AUC)

## 2) What Has Been Implemented

### 2.1 Notebook pipeline execution and data prep
- Notebook used:
  - `data/handwriting/processed/handwriting_merge_pipeline.ipynb`
- Pipeline executed end-to-end for:
  - Path setup and data checks
  - UCI/PaHaW loading and format standardization
  - Normalization and resampling
  - Full merge to timeseries CSV
  - Cleaning and summary feature extraction
  - Label merge from PaHaW metadata and UCI naming rules
- Dependency fix applied during notebook run:
  - Installed `openpyxl` for Excel metadata loading

### 2.2 SVM embedding pipeline script
- Script created:
  - `src/unimodal/handwriting/train_handwriting_svm_embeddings.py`
- Functionality:
  - Rebuilds labeled feature table from merged timeseries + metadata
  - Runs stratified 5-fold CV with PCA + SVM
  - Exports out-of-fold embeddings and predictions
  - Saves fold metrics, summary JSON, and a full-data fitted artifact

### 2.3 Multi-model benchmark script
- Script created:
  - `src/unimodal/handwriting/benchmark_handwriting_models.py`
- Models included:
  - `svm_rbf`
  - `logreg_elasticnet`
  - `gpc_rbf`
  - `random_forest`
  - `xgboost` (optional)
  - `lightgbm` (optional)
  - `catboost` (optional)
- Functionality:
  - Same stratified 5-fold split for all models
  - Per-model OOF predictions + embedding export
  - Per-model metrics and full-model artifact export
  - Global leaderboard and best-model summary

### 2.4 Warning/noise cleanup in benchmark code
- LightGBM log spam reduced by setting:
  - `verbosity=-1`
  - `force_col_wise=True`
- Feature-name mismatch warnings resolved by:
  - Using DataFrame inputs consistently for raw-feature tree models

### 2.5 SLURM scripts created and hardened
- Handwriting SLURM scripts:
  - `src/unimodal/handwriting/slurm/train_handwriting_svm_embeddings.slurm`
  - `src/unimodal/handwriting/slurm/benchmark_handwriting_models.slurm`
  - `src/unimodal/handwriting/slurm/benchmark_handwriting_models_with_installs.slurm`
- Hardening applied for RM-shared submission reliability:
  - `--mem-per-cpu=2000M` (instead of oversized total memory request)
  - RM-shared working account selection
  - logs routed to `logs/slurm/%x-%j.out/.err`

### 2.6 Final model selection pipeline (nested CV + calibration)
- Script created:
  - `src/unimodal/handwriting/finalize_handwriting_model.py`
- SLURM launcher created:
  - `src/unimodal/handwriting/slurm/finalize_handwriting_model.slurm`
- Functionality:
  - Nested CV on top candidate models (`gpc_rbf`, `random_forest`, `lightgbm` if available; fallback to `logreg_elasticnet`)
  - Hyperparameter tuning in inner CV
  - Probability calibration via `CalibratedClassifierCV`
  - Threshold selection targeting configurable recall (default 0.90)
  - Final artifact export for deployment (`final_handwriting_model.joblib`)

Current run status:
- Local login-node execution was killed due system limits (exit 137), so execution was moved to SLURM.
- SLURM finalization job completed:
  - Job ID: `38145230`
  - Name: `hw-finalize`
  - State: `COMPLETED` (exit code `0:0`)
  - Log files:
    - `logs/slurm/hw-finalize-38145230.out`
    - `logs/slurm/hw-finalize-38145230.err` (empty / clean)

Finalization outcome:
- Selected final model: `lightgbm`
- Primary metric: `roc_auc`
- Selected threshold: `0.42` (target recall `0.90`)
- Best tuned parameters:
  - `learning_rate=0.1`
  - `min_child_samples=10`
  - `n_estimators=150`
  - `num_leaves=15`
- Best inner CV AUC: `0.7569`

## 3) Current Results Snapshot

From recent benchmark run (`hw-bench-full`):
- Best model by ROC-AUC: `gpc_rbf`
- Top 5 by ROC-AUC:
  - `gpc_rbf`: 0.7898
  - `lightgbm`: 0.7850
  - `random_forest`: 0.7817
  - `catboost`: 0.7751
  - `xgboost`: 0.7615

Notes:
- A rerun confirmed clean completion with empty `.err` file for job `38144827`.
- Empty stderr in this run indicates no runtime warnings/errors emitted.

## 4) Output Artifacts

### 4.1 SVM embedding outputs
Directory:
- `outputs/unimodal_handwriting/svm_embeddings`

Files:
- `handwriting_summary_features_labeled.csv`
- `handwriting_svm_cv_metrics.csv`
- `handwriting_svm_cv_summary.json`
- `handwriting_svm_oof_embeddings.csv`
- `handwriting_svm_full_model.joblib`

### 4.2 Multi-model benchmark outputs
Directory:
- `outputs/unimodal_handwriting/model_benchmark`

Global files:
- `leaderboard.csv`
- `best_model_summary.json`
- `handwriting_summary_features_labeled.csv`

Per-model subdirectories:
- `svm_rbf/`
- `logreg_elasticnet/`
- `gpc_rbf/`
- `random_forest/`
- `xgboost/`
- `lightgbm/`
- `catboost/`

### 4.3 Final model outputs (new)
Directory:
- `outputs/unimodal_handwriting/final_model`

Expected files after `hw-finalize` completion:
Generated files (confirmed):
- `final_model_leaderboard.csv`
- `all_models_oof_predictions.csv`
- `final_model_summary.json`
- `final_handwriting_model.joblib`
- `final_model_input_table.csv`
- `<model_name>/nested_cv_fold_metrics.csv`
- `<model_name>/best_params_per_fold.json`
- `<model_name>/oof_predictions.csv`

## 5) Runbook

### 5.1 Local SVM embeddings
From project root:
- `conda run -n chiu-lab python src/unimodal/handwriting/train_handwriting_svm_embeddings.py --n-splits 5 --embedding-dim 4`

### 5.2 Local full benchmark
From project root:
- `conda run -n chiu-lab python src/unimodal/handwriting/benchmark_handwriting_models.py --n-splits 5 --embedding-dim 4 --primary-metric roc_auc`

### 5.3 SLURM benchmark with optional package install
From project root:
- `sbatch src/unimodal/handwriting/slurm/benchmark_handwriting_models_with_installs.slurm`

### 5.4 SLURM logs location
- `logs/slurm/<job-name>-<job-id>.out`
- `logs/slurm/<job-name>-<job-id>.err`

### 5.5 Finalize deployable model (SLURM)
From project root:
- `sbatch src/unimodal/handwriting/slurm/finalize_handwriting_model.slurm`

## 6) Recommended Next Steps

1. Perform nested CV for top models (`gpc_rbf`, `random_forest`, `lightgbm`) to reduce selection bias.
2. Add probability calibration and choose an operating threshold by clinical objective (for example high recall target).
3. Add a compact error-analysis report by dataset source (UCI vs PaHaW) and by false-negative group.
4. Freeze a final inference package containing feature order, scaler/PCA/model artifact, threshold, and label mapping.

## 7) Change Log

- 2026-03-22:
  - Executed handwriting merge notebook pipeline end-to-end.
  - Added SVM embedding script and validated outputs.
  - Added multi-model benchmark script and SLURM launchers.
  - Installed optional tree-boosting libraries in benchmark run and completed full comparison.
  - Fixed RM-shared submission constraints (memory/account) for handwriting SLURM jobs.
  - Fixed benchmark warning/noise issues (LightGBM verbosity + feature-name consistency).
  - Added this handwriting progress README.
- 2026-03-22 (late):
  - Added nested-CV finalization pipeline (`finalize_handwriting_model.py`).
  - Added SLURM launcher for final model selection (`finalize_handwriting_model.slurm`).
  - Submitted and completed finalization run on RM-shared (`hw-finalize`, job `38145230`).
  - Final selected model: `lightgbm` with threshold `0.42` for target recall `0.90`.
