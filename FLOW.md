# Flow

How the code runs, one section per version. Entry points and execution order only. Git history records what changed. Update this file only when the flow itself changes.

Every item below was read from the code on 2026-10-09. None of the commands were run to write this file. Where the repo does not record something, this file says so.

## v1 (in `v1/`, frozen)

Run v1 commands from inside `v1/`. Each script finds `data/` and `outputs/` relative to its own location (`Path(__file__)` plus a fixed number of parent folders), so the folder can sit anywhere. The cluster launchers (`*.slurm`) also contain absolute paths from the original course environment and will not run elsewhere unchanged.

### Late fusion and explainability

1. `python -m src.multimodal_fusion.evaluate [--strategy equal|auc_weighted|softmax_auc_weighted|confidence_weighted] [--no-save]`. The default strategy is `auc_weighted`.
2. `evaluate.py` calls `loaders.load_all_from_embeddings()`, which reads from `src/multimodal_fusion/embeddings/`. Which file it reads: handwriting is pinned to `handwriting_embeddings_0406.csv`; gait is the last file by name among `gait_embeddings_*.npz` and `*.csv`; speech is the last match for `speech_embeddings_*.npz`. Adding a file with a later name changes the gait or speech input.
3. It fits `LateFusionModel(strategy, calibrate=True)` from `fusion.py`, computes a metrics table with bootstrap confidence intervals, and writes `outputs/multimodal_fusion/fusion_model.json` and `metrics_table.csv`.
4. The three embedding sets come from different people. The fusion scores are estimated by randomly pairing subjects across modalities and labeling each pair by majority vote (`fusion.py`: the bootstrap evaluation of `LateFusionModel` and `StackingFusionModel`). They are simulated results, not measurements on people who have every modality.
5. `explainability.py` defines `FusionExplainer` and `GaitEmbeddingExplainer` (SHAP). No script, notebook or launcher in the repo calls them. How the SHAP outputs were produced is not recorded.
6. The repo does not record which script wrote each committed file in `src/multimodal_fusion/embeddings/`.

### Gait: figshare IMU data (severity among PD patients)

- `python src/unimodal/gait/train_gait.py` takes no command-line flags. A temporal convolutional network (TCN) separates Mild PD (Hoehn and Yahr 2.0 or lower) from Moderate/Severe PD (above 2.0), using freezing-of-gait patients only. This is a severity task, not PD against healthy controls. It reads `data/gait/figshare/IMU/` and `PDFEinfo.csv`, uses `StratifiedGroupKFold` with 5 folds and seed 42, and writes `outputs/unimodal_gait/PDFE_Severity_Classification/` (`cv_results.csv`, `predictions.npz`, `scaler.pkl`, `summary.json`, one folder per fold).
- `python src/unimodal/gait/train_gait_rf.py` is a random forest baseline on engineered features for the same severity task.

### Gait: WearGait-PD

1. `python src/unimodal/gait/prepare_weargait_index.py [--data-root PATH] [--out-csv PATH]` builds an index of the WearGait files, with group 1 for paths under `PD PARTICIPANTS` and 0 otherwise. Default output: `outputs/unimodal_gait/weargait_index.csv`.
2. `python src/unimodal/gait/train_weargait_embeddings.py --tasks <task>` trains a model for one walking task and writes subject embeddings and predictions under `outputs/unimodal_gait/weargait_dl_embeddings/<task>/` (`weargait_subject_embeddings.npz`, `predictions.npz`, `cv_metrics.csv`, `summary.json`, one model file per fold). The tasks are SelfPace, HurriedPace and TUG.
3. `python src/unimodal/gait/gait_ensemble_orchestrator.py --tasks <pdfe,weargait,rf or all> [--force]` runs, in order: `train_gait.py`, step 2 for each walking task, `concat_weargait_task_embeddings.py`, then `train_gait_rf.py`. It writes `outputs/unimodal_gait/ensemble_summary.json`. **`concat_weargait_task_embeddings.py` is not tracked on `main`.** It was deleted on 2026-03-26 in `8207708` and can be restored from `be97d0b`. Without it, the WearGait branch of the orchestrator fails at that step.
4. `python src/unimodal/gait/ensemble_fusion.py --strategy all` combines the saved prediction files from the three gait tasks (weighted average, stacking, calibrated late fusion, voting) where their labels align.
5. Experiments: `src/unimodal/gait/experiments/weargait_representation`, `weargait_multimodal_compare` and `weargait_update3_ablation`. Each has a `run_experiment.py` and a Slurm launcher, and works under `outputs/unimodal_gait/`. `benchmark_weargait_representations.py` also reads there. These were not examined in detail.
6. Cluster launchers: `src/unimodal/gait/slurm/` runs the orchestrator (`--tasks all --force`, or `--tasks rf --force` on CPU) and then `ensemble_fusion.py --strategy all`. `train_gait.slurm` and `train_weargait_embeddings.slurm` (which runs steps 1 and 2) sit beside the scripts.

### Handwriting

Scripts in `src/unimodal/handwriting/`, each with a Slurm launcher in `slurm/`:

- `train_handwriting_svm_embeddings.py` writes out-of-fold predictions, cross-validation metrics and a feature table. The README describes it as an SVM that exports drawing-level embeddings.
- `benchmark_handwriting_models.py [--n-splits N] [--embedding-dim N] [--seed N]` cross-validates several classifiers. It writes per-model metrics and out-of-fold embeddings, a leaderboard, and `handwriting_summary_features_labeled.csv`.
- `finalize_handwriting_model.py [--outer-splits N] [--inner-splits N] [--primary-metric M] [--target-recall X] [--seed N]` runs nested cross-validation, ranks the models, and writes `all_models_oof_predictions.csv` and `final_model_input_table.csv`.
- The `in_air_*` pipelines and `eda_non_linear_relationships.py` were not examined.

Default output folders are computed from each script's location. The input paths, and which script wrote each committed `handwriting_embeddings_*.csv`, are not recorded.

### Speech

`src/unimodal/speech/scripts/TrainSpeechBasedModel_v2.0.ipynb` trains a CNN on mel-spectrograms and a CatBoost model, and calls `np.savez`. It reads precomputed mel-spectrogram files from absolute paths under a home directory on another machine, and nothing in it computes mel-spectrograms. The code that made those files is not in the repo, so speech cannot be rerun from here. The notebook never uses the name `speech_embeddings`, so which code wrote the committed `speech_embeddings_0325.npz` is not recorded. Utility scripts in the same folder: `sort_by_diagnosis.py`, `sort_by_task.py`, `sort_static.py`, `findStaticOutliers.py`, `tryTrainForStaticFeatures.py`. An older notebook is at `notebooks/modelTraining/TrainSpeechBasedModel.ipynb`.

### Other

- `notebooks/eda/` holds exploratory notebooks for gait, handwriting and WearGait-PD, and speech quality-control outputs.
- `scripts/verify_setup.py` checks that a set of folders and files exists. Its list describes an earlier layout of the project.
- `tests/test_download_weargait.py` (standard library `unittest`) checks the WearGait download script and manifests. Run it from the repo root with `python3 -m unittest discover -s v1/tests -v`.

## v2 (in `v2/`)

No code yet. See `v2/README.md` for the plan. Add its entry points and execution order here when they exist.
