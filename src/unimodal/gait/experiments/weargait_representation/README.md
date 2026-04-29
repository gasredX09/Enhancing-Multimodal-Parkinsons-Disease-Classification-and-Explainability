# WearGait Representation Benchmark

This experiment compares the current WearGait subject-level representations on the common subject intersection:
- `SelfPace`
- `HurriedPace`
- `TUG`
- concatenated gait embedding (`Concat`)

## What it does
- Loads the saved subject embeddings from the latest WearGait task-aware training outputs
- Aligns all representations to the common subject intersection
- Runs repeated stratified CV with logistic regression on top of each embedding
- Writes a run-named output directory under `outputs/unimodal_gait/runs/`
- Saves per-split metrics, summary metrics, a benchmark plot, and notes

## Python entrypoint
```bash
python src/unimodal/gait/experiments/weargait_representation/run_experiment.py \
  --run-dir outputs/unimodal_gait/runs/<run_name>
```

## SLURM entrypoint
```bash
sbatch src/unimodal/gait/experiments/weargait_representation/slurm/benchmark_rm_shared.slurm
```

Override the run name at submit time:
```bash
sbatch --export=ALL,RUN_NAME=weargait_representation_benchmark_<tag> \
  src/unimodal/gait/experiments/weargait_representation/slurm/benchmark_rm_shared.slurm
```
