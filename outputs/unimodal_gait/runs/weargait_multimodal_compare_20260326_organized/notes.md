# WearGait Downstream Multimodal Comparison

Run directory: `project/outputs/unimodal_gait/runs/weargait_multimodal_compare_20260326_organized`

## What this run does
- Builds subject-level gait probabilities from the current `TUG` and `Concat` embeddings
- Uses repeated stratified CV logistic regression on the gait embeddings
- Compares downstream multimodal late fusion using the same handwriting and speech inputs
- Evaluates four fusion strategies from `multimodal_fusion.fusion.LateFusionModel`

## Inputs used
- Handwriting model: `best`
- Speech model: `mean`
- Gait candidates: `TUG`, `Concat`

## Important caveat
- These multimodal fusion results are simulation-based because the modality cohorts are not the same subjects.
- The fusion AUCs are useful for relative comparison between `TUG` and `Concat`, but they are not a substitute for a true aligned multimodal cohort evaluation.

## Key takeaways
- Best downstream fusion configuration by simulated AUC: `TUG + softmax_auc_weighted`
- Best gait-only input by unimodal AUC in this run: `TUG` (0.811)
- The more promising gait branch for the next multimodal phase is whichever stays stronger across both unimodal and fusion views.

## Fusion strategy ranking
gait_representation             strategy  auc_mean  auc_ci_lo  auc_ci_hi
                TUG softmax_auc_weighted  0.727547   0.583323   0.855905
                TUG         auc_weighted  0.720783   0.571636   0.855922
             Concat softmax_auc_weighted  0.715966   0.571422   0.852273
                TUG                equal  0.712611   0.562500   0.849359
             Concat         auc_weighted  0.708955   0.556076   0.846685
             Concat                equal  0.700629   0.548439   0.836545
                TUG  confidence_weighted  0.696189   0.544978   0.837021
             Concat  confidence_weighted  0.676338   0.524038   0.819675

## Modality metrics
gait_representation    modality      auc       f1  accuracy   weight
                TUG        gait 0.810794 0.780488  0.751381 0.364162
                TUG handwriting 0.568318 0.582278  0.547945 0.255256
                TUG      speech 0.847354 0.777778  0.779310 0.380583
             Concat        gait 0.784977 0.760417  0.745856 0.356702
             Concat handwriting 0.568318 0.582278  0.547945 0.258250
             Concat      speech 0.847354 0.777778  0.779310 0.385047

## Recommended next step
- Carry the stronger of `TUG` vs `Concat` forward as the primary gait input for multimodal fusion experiments.
- Keep the other as an ablation so the presentation can show that the choice was tested rather than assumed.
