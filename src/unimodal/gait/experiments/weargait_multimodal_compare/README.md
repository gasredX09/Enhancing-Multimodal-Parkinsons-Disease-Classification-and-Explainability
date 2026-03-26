# WearGait Downstream Multimodal Comparison

This experiment compares the two strongest current gait candidates for downstream use:
- `TUG`
- `Concat` (SelfPace + HurriedPace + TUG embeddings concatenated)

## What it does
- Converts each gait embedding into subject-level out-of-fold probabilities using repeated stratified CV logistic regression.
- Reuses the current handwriting and speech OOF predictions.
- Runs the late-fusion comparison from `src/multimodal_fusion/fusion.py`.
- Writes a fresh run directory under `project/outputs/unimodal_gait/runs/`.

## Important caveat
The fusion module is simulation-based because gait, handwriting, and speech do not currently share a single aligned subject cohort. Use this run to compare `TUG` versus `Concat` under the same downstream setup, not to claim a final real-world multimodal AUC.

## Local run
```bash
python project/src/unimodal/gait/experiments/weargait_multimodal_compare/run_experiment.py \
  --run-dir project/outputs/unimodal_gait/runs/weargait_multimodal_compare_YYYYMMDD_HHMMSS
```

## SLURM run
```bash
RUN_NAME=weargait_multimodal_compare_YYYYMMDD_HHMMSS \
  sbatch project/src/unimodal/gait/experiments/weargait_multimodal_compare/slurm/benchmark_rm_shared.slurm
```

## Outputs
- `gait_repeated_cv_split_metrics.csv`
- `gait_oof_probabilities.csv`
- `modality_metrics.csv`
- `fusion_strategy_summary.csv`
- `fusion_detailed_results.json`
- `fusion_strategy_comparison.png`
- `modality_weight_comparison.png`
- `notes.md`
- `run_summary.json`
