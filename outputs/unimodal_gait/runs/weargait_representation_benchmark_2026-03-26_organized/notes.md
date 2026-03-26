# WearGait Representation Benchmark Notes

Run directory: `project/outputs/unimodal_gait/runs/weargait_representation_benchmark_2026-03-26_organized`

## What this run does
- Compares `SelfPace`, `HurriedPace`, `TUG`, and concatenated gait embeddings
- Uses the common subject intersection across all representations
- Uses repeated stratified CV with logistic regression on top of each embedding

## Key result
- Best representation by AUC: `TUG`
- Mean AUC: `0.810`
- Mean F1: `0.767`
- Common subject count: `181`

## Interpretation
- `TUG` is currently the strongest standalone gait representation.
- The concatenated embedding is still competitive and remains useful as a richer fusion-ready representation.
- The most meaningful next comparison is `TUG` vs `Concat` in downstream fusion.

## Ranking by AUC
representation  mean_auc  mean_f1  mean_accuracy
           TUG  0.809687 0.766652       0.742748
        Concat  0.769992 0.730438       0.712763
      SelfPace  0.701094 0.681129       0.645961
   HurriedPace  0.632462 0.594724       0.580195

## Recommended next step
- Carry both `TUG` and `Concat` forward as candidate gait inputs for downstream fusion.
