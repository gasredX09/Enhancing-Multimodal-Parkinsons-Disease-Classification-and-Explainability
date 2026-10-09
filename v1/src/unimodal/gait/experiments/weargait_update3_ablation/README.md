# WearGait Update-3 Gait-Only Ablation

This experiment packages the highest-value remaining gait-only Update-3 work:
- task ablations: `TUG`, `SelfPace`, `SelfPace+TUG`, `All3`, `HurriedPace`
- downstream unimodal classifiers on subject embeddings
- reduced-overfitting comparisons with regularized models and dimensionality reduction
- repeated-CV reporting with confidence intervals

## What it runs
- Representation inputs:
  - `TUG`
  - `SelfPace`
  - `SelfPace+TUG`
  - `All3` (concatenated `SelfPace + HurriedPace + TUG`)
  - `HurriedPace`
- Model variants:
  - `lr_l2`
  - `lr_l1`
  - `pca64_lr_l2`
  - `pca32_lr_l2`
  - `mi128_lr_l2`

## Output policy
This experiment requires a fresh run directory and will fail rather than overwrite an existing one.

## Local run
```bash
python project/src/unimodal/gait/experiments/weargait_update3_ablation/run_experiment.py \
  --run-dir project/outputs/unimodal_gait/runs/weargait_update3_ablation_YYYYMMDD_HHMMSS
```

## SLURM run
```bash
RUN_NAME=weargait_update3_ablation_YYYYMMDD_HHMMSS \
  sbatch project/src/unimodal/gait/experiments/weargait_update3_ablation/slurm/benchmark_rm_shared.slurm
```

## Outputs
- `per_split_metrics.csv`
- `summary_metrics.csv`
- `representation_model_auc_heatmap.png`
- `top_configurations_auc.png`
- `notes.md`
- `run_summary.json`
