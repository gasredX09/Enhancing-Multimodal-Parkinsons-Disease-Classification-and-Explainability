# WearGait Update-3 Gait-Only Ablation Notes

Run directory: `/ocean/projects/med260006p/aguda1/biomedAI/project/outputs/unimodal_gait/runs/weargait_update3_ablation_20260406_175546`

## What this run does
- Compares task-level gait representations: `TUG`, `SelfPace`, `SelfPace+TUG`, `All3`, and `HurriedPace`
- Evaluates regularized downstream classifiers and dimensionality-reduction variants
- Uses repeated stratified CV for more stable gait-only reporting

## Subject coverage
- Common all-task cohort: `181`
- Common SelfPace+TUG cohort: `181`

## Best overall configuration
- Representation: `TUG`
- Model: `pca32_lr_l2`
- Mean AUC: `0.810`
- Mean F1: `0.768`
- 95% AUC interval: `0.690` to `0.912`

## Best model per representation
representation       model  mean_auc  mean_f1  mean_accuracy
          All3 mi128_lr_l2  0.801465 0.740800       0.728784
   HurriedPace       lr_l1  0.650320 0.591671       0.600541
      SelfPace       lr_l1  0.732158 0.718870       0.684084
  SelfPace+TUG mi128_lr_l2  0.799056 0.731063       0.714505
           TUG pca32_lr_l2  0.810054 0.768062       0.744384

## Interpretation
- This run directly addresses the remaining gait-only Update-3 questions: task ablation, regularized downstream evaluation, and whether reduced-dimensional models help.
- The recommended gait branch for the next internal baseline should be the strongest representation-model pair from this table.

## Run settings
- n_splits: `5`
- n_repeats: `10`
- seed: `42`
