# WearGait Update Notes

## Slide 1: What changed in gait
- We moved from a single shared WearGait model to task-aware modeling.
- We trained one separate TCN encoder for each task: `SelfPace`, `HurriedPace`, and `TUG`.
- We then concatenated the three subject-level embeddings to create one fusion-ready gait representation.

Suggested talk track:
The main change in the gait branch was to stop forcing all three walking protocols into one shared encoder. Each task now learns its own temporal representation, and those task-specific embeddings are combined afterward.

## Slide 2: Separate-task results
Use:
- `weargait_task_metrics_bar.png`
- `weargait_fold_heatmaps.png`

Key points:
- `TUG` is the strongest single task by AUC at about `0.749`.
- `SelfPace` is the most balanced task overall, with the best F1 at about `0.754`.
- `HurriedPace` is weaker than the other two alone, but may still contribute complementary signal when combined.
- The fold heatmaps show that the task-specific models are learning signal consistently, although there is still fold-to-fold variance.

Suggested talk track:
The important observation is that the three tasks are not equally informative. TUG appears to capture the strongest discriminative signal, while SelfPace remains a strong balanced baseline. That supports the idea of learning them separately rather than averaging them away in one shared model.

## Slide 3: Data cleanup and coverage recovery
Use:
- `weargait_coverage_recovery.png`

Key points:
- After re-downloading WearGait, the indexed cohort reached `185` subjects.
- We audited the drop in usable files and found that most losses came from missing force / center-of-pressure channels.
- We changed the required channel set to an IMU-only 18-channel core.
- That improved the final concatenated cohort from `170` subjects to `181` subjects.

Suggested talk track:
This was the main engineering fix. The earlier channel requirement was too strict for the real WearGait files. By keeping the shared IMU channels and removing the most inconsistent channels, we recovered more subjects without throwing away the core gait signal.

## Slide 4: Fusion-ready gait representation
Use:
- `weargait_fusion_ready_summary.png`
- `weargait_summary_slide.png`

Key points:
- Each task produces a `256`-dimensional subject embedding.
- Concatenating the three tasks gives a `768`-dimensional gait embedding.
- The final fusion-ready gait embedding covers `181` subjects.
- This is the final output of the current gait stage.

Suggested talk track:
The output we care about at this stage is the final subject-level gait representation. We now have a task-aware, cleaned, fusion-ready embedding that carries complementary information from all three walking protocols.

## Slide 5: Architecture note
- All three tasks currently use the same backbone: `TCN`.
- We did not use different architectures per task in this round.
- Keeping the same backbone made the comparison interpretable and kept the story clean.
- A later ablation can test whether `TUG` benefits from a different architecture, but that is a follow-up experiment rather than the current result.

Suggested talk track:
We intentionally kept the same architecture across all three tasks so that the main experimental difference was the task split itself. That makes the current result easier to interpret.

## Slide 6: Next steps for gait
- Use the current `181 x 768` task-aware gait embedding as the main gait input for downstream fusion.
- Compare the concatenated gait embedding against single-task gait baselines.
- Run ablations: `SelfPace` only, `TUG` only, and all three tasks concatenated.
- Add repeated grouped cross-validation and confidence intervals for more stable reporting.
- Test targeted architecture ablations later, with `TCN` as the reference baseline.

Suggested talk track:
The gait branch is now in a stable place. The next step is to use this representation in the downstream fusion setup, then come back and test which individual tasks and future architecture changes matter most.

## Recommended one-slide summary
- Re-downloaded and restored the full WearGait cohort: `185` subjects indexed
- Switched from one shared model to separate TCN encoders for `SelfPace`, `HurriedPace`, and `TUG`
- Replaced the overly strict 24-channel requirement with an 18-channel IMU-only core set
- Improved the usable concatenated gait cohort from `170` to `181`
- Best single-task AUC came from `TUG` (`~0.749`)
- Final output: fusion-ready gait embedding with `181` subjects and `768` dimensions
