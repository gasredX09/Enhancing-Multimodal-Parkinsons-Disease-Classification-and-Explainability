# Gait Progress README

Last updated: 2026-03-22
Owner: Copilot-assisted implementation
Scope: Unimodal gait pipeline (TCN + WearGait TCN + Random Forest + Fusion)

## 1) Objective

Build a robust gait modeling pipeline for Parkinson severity classification with:
- A deep baseline on figshare IMU windows (TCN)
- A wearable-sensor TCN representation model (WearGait)
- A handcrafted-feature baseline (Random Forest)
- Subject-aligned ensemble fusion across all gait tasks

This README tracks what has been implemented, where outputs are stored, what is still pending, and the recommended execution order.

## 2) Current Pipeline Architecture

### 2.1 Task A: PDFE Severity Classification (TCN)
- Script: src/unimodal/gait/train_gait.py
- Data source: data/gait/figshare
- Model: TCN classifier over sliding windows
- CV strategy: StratifiedGroupKFold
- Output dir: outputs/unimodal_gait/PDFE_Severity_Classification

### 2.2 Task B: WearGait Task Embeddings (TCN)
- Script: src/unimodal/gait/train_weargait_embeddings.py
- Data source: outputs/unimodal_gait/weargait_index.csv -> WearGait CSV files
- Model: TCN with embedding head + classification head
- CV strategy: StratifiedGroupKFold
- Output dir: outputs/unimodal_gait/WearGait_Task_Embeddings

### 2.3 Task C: Random Forest Baseline
- Script: src/unimodal/gait/train_gait_rf.py
- Data source: data/gait/figshare/IMU + labels
- Model: RandomForestClassifier on engineered features
- CV strategy: StratifiedGroupKFold
- Output dir: outputs/unimodal_gait/Random_Forest_Baseline

### 2.4 Fusion
- Script: src/unimodal/gait/ensemble_fusion.py
- Input artifacts: predictions.npz from all three tasks
- Strategies: weighted averaging, stacking, voting
- Output dir: outputs/unimodal_gait/fusion_results

### 2.5 Orchestration
- Script: src/unimodal/gait/gait_ensemble_orchestrator.py
- Role: run tasks consistently and generate ensemble summary

## 3) What Was Implemented In This Update

### 3.1 Standardized output schema for fusion
A common predictions artifact is now used by all tasks:
- File name: predictions.npz
- Required keys:
  - subject_ids
  - y_true
  - y_pred
  - y_proba (2 columns: P(class 0), P(class 1))

This enables deterministic subject-level fusion across heterogeneous model types.

### 3.2 train_gait.py updated for task-specific outputs + OOF subject predictions
Changes made:
- Output moved to task folder: outputs/unimodal_gait/PDFE_Severity_Classification
- Saved scaler to task folder
- Added OOF collection during CV
- Aggregated window-level OOF probabilities to subject-level predictions
- Saved predictions.npz with both subject-level and window-level outputs
- Extended summary.json with output metadata

New/updated artifacts in Task A folder:
- summary.json
- cv_results.csv
- scaler.pkl
- predictions.npz
- fold_*/ (best model, confusion matrix, classification report)

### 3.3 train_weargait_embeddings.py updated to export predictions.npz
Changes made:
- Added infer_predictions helper to generate probability and class predictions
- Collected OOF fold predictions at window level
- Aggregated to subject-level probability and prediction
- Saved predictions.npz in task output dir
- Updated summary.json outputs block to include predictions_npz

New/updated artifacts in Task B folder:
- summary.json
- cv_metrics.csv
- weargait_subject_embeddings.npz
- predictions.npz
- fold_*_best_model.pt

### 3.4 train_gait_rf.py updated with subject IDs in predictions
Changes made:
- Added subject_ids to OOF prediction artifact
- Enriched summary metadata with output paths and prediction level

New/updated artifacts in Task C folder:
- summary.json
- cv_results.csv
- predictions.npz
- rf_models.pkl
- confusion_matrix.png

### 3.5 gait_ensemble_orchestrator.py aligned with task-specific WearGait output
Changes made:
- WearGait task now runs with explicit --output-dir outputs/unimodal_gait/WearGait_Task_Embeddings

This removes location mismatch between training and fusion expectations.

### 3.6 ensemble_fusion.py hardened for real-world multi-task alignment
Changes made:
- Replaced wildcard loading with explicit known task folders:
  - PDFE_Severity_Classification
  - WearGait_Task_Embeddings
  - Random_Forest_Baseline
- Added schema validation for prediction files
- Added fallback for 1D probabilities -> converted to 2-column probabilities
- Added alignment by subject ID intersection across tasks
- Added label mismatch diagnostics after alignment

Result:
- Fusion now works on aligned subjects only, reducing silent leakage/misalignment risk.

## 4) Execution Runbook

### 4.1 Train all gait tasks
From project root:
python src/unimodal/gait/gait_ensemble_orchestrator.py --tasks all

### 4.2 Run fusion comparison
python src/unimodal/gait/ensemble_fusion.py --strategy all

### 4.3 Expected final outputs
- outputs/unimodal_gait/ensemble_summary.json
- outputs/unimodal_gait/fusion_results/fusion_comparison.json
- outputs/unimodal_gait/fusion_results/fusion_comparison.png

## 5) Current Known Status

Based on existing summaries observed before this update:
- PDFE TCN showed strong F1 but weak AUC in prior run (likely calibration or score orientation issue)
- WearGait had moderate balanced metrics
- RF baseline now exists and is integrated structurally

Important note:
After this code update, re-running tasks is required to regenerate new predictions.npz artifacts in the updated schema and paths.

## 6) Validation Checklist (Post-Run)

After running orchestrator and fusion:
- Confirm all three files exist:
  - outputs/unimodal_gait/PDFE_Severity_Classification/predictions.npz
  - outputs/unimodal_gait/WearGait_Task_Embeddings/predictions.npz
  - outputs/unimodal_gait/Random_Forest_Baseline/predictions.npz
- Confirm fusion reports:
  - aligned common subject count > 0
  - no severe label mismatch across tasks
- Confirm strategy metrics are sensible:
  - accuracy and F1 improve or are at least stable vs best single model
  - AUC is not degenerate

## 7) Recommended Next Steps (Priority Order)

### Priority 1: Regenerate all artifacts with the new schema
- Re-run all three tasks and fusion once end-to-end
- Freeze resulting metrics in a versioned experiment log

### Priority 2: Metric integrity checks
- Verify label orientation for AUC in every task
- Inspect probability calibration curves for TCN and RF
- If needed, calibrate with Platt or isotonic at subject level

### Priority 3: Better fusion training protocol
- Move stacking to true nested CV / OOF-only meta-learning
- Avoid fitting meta-learner on the same labels used for base model generation

### Priority 4: Threshold tuning by clinical objective
- Define target operating point (high recall vs balanced)
- Tune threshold on validation folds at subject level
- Report decision-threshold sensitivity

### Priority 5: Feature/architecture experiments
- RF: tune depth, min samples, and feature subsets
- TCN: window length sweep, dilation schedule sweep, dropout sweep
- WearGait: task subset ablations (SelfPace vs TUG etc.)

### Priority 6: Explainability and interpretability
- RF: SHAP/permutation importance at subject level
- TCN: saliency/temporal attribution by channel and segment
- Compare agreement between feature-based and deep feature importance

### Priority 7: Reporting package
- Build one table with per-task and fusion metrics:
  - Accuracy, Precision, Recall, F1, AUC, N-subjects
- Save confusion matrices and key failure cases by subject/task

## 8) Suggested Weekly Maintenance Routine

At each gait iteration:
1. Run selected tasks (or all)
2. Run fusion
3. Append results snapshot to this README
4. Record what changed (data, hyperparameters, code)
5. Record what improved and what regressed
6. Set next 1-3 experiments

## 9) Change Log

- 2026-03-22:
  - Added standardized predictions.npz schema across gait tasks
  - Added subject-level OOF export in PDFE TCN and WearGait TCN
  - Added subject_ids in RF predictions artifact
  - Updated orchestrator output routing for WearGait task
  - Hardened fusion loader with explicit task dirs + subject alignment
  - Created this detailed progress README

## 10) Bridges2 SLURM Run Plan (Added)

To support cluster runs with resource constraints, two production scripts were added:

- src/unimodal/gait/slurm/gait_pipeline_gpu_shared.slurm
  - Intended use: full gait pipeline (TCN + WearGait + RF + fusion)
  - Partition: GPU-shared
  - Default resources: 1x v100-32 GPU, 8 CPU cores, 64G RAM, 24h walltime
  - Includes runtime guard for GPU-shared policy (max 4 GPUs)
  - Includes conda activation fallback logic and environment diagnostics

- src/unimodal/gait/slurm/gait_pipeline_cpu_rm_shared.slurm
  - Intended use: RF + fusion when GPU is not required or not available
  - Partition: RM-shared
  - Default resources: 16 CPU cores, 64G RAM, 24h walltime

### 10.1 Submission commands

From project root:

- Full GPU pipeline:
  - sbatch src/unimodal/gait/slurm/gait_pipeline_gpu_shared.slurm

- CPU-only baseline + fusion:
  - sbatch src/unimodal/gait/slurm/gait_pipeline_cpu_rm_shared.slurm

### 10.2 Limits used for script design

Using Bridges2 user guide constraints (GPU and GPU-shared sections):

- GPU-shared allows at most 4 GPUs per job
- GPU-shared max walltime is 48h
- RM-shared allows one node and up to 64 cores

Scripts are configured conservatively to fit those limits and to reduce pending time.

### 10.3 Operational notes

- If your default conda env is not `chiu-lab`, submit with:
  - sbatch --export=ALL,CONDA_ENV=<env_name> src/unimodal/gait/slurm/gait_pipeline_gpu_shared.slurm
- Logs are written to:
  - project/logs/slurm/<job-name>-<job-id>.out/.err
- To inspect partition availability before submit:
  - slurm-tool p
  - sinfo -p GPU-shared
