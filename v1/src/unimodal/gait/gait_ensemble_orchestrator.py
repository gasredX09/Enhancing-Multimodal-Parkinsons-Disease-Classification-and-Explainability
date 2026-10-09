"""
Gait Ensemble Orchestrator
========================

Unified training pipeline for 3 independent gait tasks:
1. Task 1: PDFE Severity Classification (figshare) - calls train_gait.py
2. Task 2: WearGait task-specific embeddings - trains SelfPace, HurriedPace, and TUG separately
3. Task 3: Random Forest Baseline (engineered features) - calls train_gait_rf.py

Each task is trained independently with proper stratified cross-validation.
Results and predictions are saved separately for later fusion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


TASK_CONFIGS = {
    "pdfe": {
        "name": "PDFE_Severity_Classification",
        "script": "train_gait.py",
        "description": "Temporal Convolutional Network on figshare IMU data",
    },
    "weargait": {
        "name": "weargait_dl_embeddings",
        "script": "train_weargait_embeddings.py",
        "description": "Task-specific WearGait TCN encoders plus concatenated subject embeddings",
    },
    "rf": {
        "name": "Random_Forest_Baseline",
        "script": "train_gait_rf.py",
        "description": "Random Forest with engineered gait features",
    },
}

WEARGAIT_TASKS = ["SelfPace", "HurriedPace", "TUG"]


class EnsembleOrchestrator:
    def __init__(self, project_root: Path, device: str = "cuda", seed: int = 42):
        self.project_root = Path(project_root)
        self.device = device
        self.seed = seed

        self.src_dir = self.project_root / "src" / "unimodal" / "gait"
        self.data_dir = self.project_root / "data" / "gait"
        self.output_dir = self.project_root / "outputs" / "unimodal_gait"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Dict] = {}

        print("✓ Initialized EnsembleOrchestrator")
        print(f"  Script dir: {self.src_dir}")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Output dir: {self.output_dir}")

    def run_command(self, cmd: List[str], description: str) -> bool:
        print(f"\n{'=' * 60}")
        print(description)
        print(f"{'=' * 60}")
        print(f"Command: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True, text=True)
            print(f"\n✓ {description} COMPLETED\n")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"\n✗ {description} FAILED")
            print(f"Error: {exc}\n")
            return False

    def train_task(self, task_name: str, force: bool = False) -> Dict:
        if task_name == "pdfe":
            return self._train_pdfe_severity(force)
        if task_name == "weargait":
            return self._train_weargait_embeddings(force)
        if task_name == "rf":
            return self._train_rf_baseline(force)
        raise ValueError(f"Unknown task: {task_name}")

    def _train_pdfe_severity(self, force: bool = False) -> Dict:
        task_output_dir = self.output_dir / "PDFE_Severity_Classification"
        if (task_output_dir / "summary.json").exists() and not force:
            print("\n✓ PDFE results found, skipping training.")
            with open(task_output_dir / "summary.json") as f:
                return json.load(f)

        cmd = [sys.executable, str(self.src_dir / "train_gait.py")]
        success = self.run_command(cmd, "TASK 1: PDFE Severity Classification (TCN)")

        result = {
            "task": "pdfe_severity",
            "script": "train_gait.py",
            "status": "completed" if success else "failed",
        }
        if success:
            summary_file = task_output_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    result.update(json.load(f))

        self.results["pdfe"] = result
        return result

    def _train_weargait_embeddings(self, force: bool = False) -> Dict:
        base_output_dir = self.output_dir / "weargait_dl_embeddings"
        concat_output_dir = self.output_dir / "weargait_concat_embeddings"
        concat_summary = concat_output_dir / "summary.json"

        if concat_summary.exists() and not force:
            print("\n✓ Concatenated WearGait embeddings found, skipping training.")
            with open(concat_summary) as f:
                result = json.load(f)
            result.setdefault("status", "completed")
            self.results["weargait"] = result
            return result

        task_summaries: Dict[str, Dict] = {}
        all_success = True

        for task in WEARGAIT_TASKS:
            task_output_dir = base_output_dir / task
            summary_file = task_output_dir / "summary.json"
            if summary_file.exists() and not force:
                print(f"\n✓ WearGait {task} results found, skipping training.")
                with open(summary_file) as f:
                    task_summaries[task] = json.load(f)
                continue

            cmd = [
                sys.executable,
                str(self.src_dir / "train_weargait_embeddings.py"),
                "--tasks",
                task,
                "--output-dir",
                str(task_output_dir),
            ]
            success = self.run_command(cmd, f"TASK 2: WearGait {task} Encoder (TCN)")
            all_success = all_success and success

            if success and summary_file.exists():
                with open(summary_file) as f:
                    task_summaries[task] = json.load(f)

        if all_success:
            concat_cmd = [
                sys.executable,
                str(self.src_dir / "concat_weargait_task_embeddings.py"),
                "--input-root",
                str(base_output_dir),
                "--output-dir",
                str(concat_output_dir),
                "--tasks",
                *WEARGAIT_TASKS,
            ]
            all_success = self.run_command(
                concat_cmd,
                "TASK 2B: Concatenate WearGait task embeddings",
            )

        result: Dict[str, object] = {
            "task": "weargait_task_specific_embeddings",
            "script": "train_weargait_embeddings.py + concat_weargait_task_embeddings.py",
            "status": "completed" if all_success else "failed",
            "task_summaries": task_summaries,
            "task_outputs": {task: str(base_output_dir / task) for task in WEARGAIT_TASKS},
            "concat_output_dir": str(concat_output_dir),
        }

        if concat_summary.exists():
            with open(concat_summary) as f:
                concat_data = json.load(f)
            result.update({
                "concat_summary": concat_data,
                "concatenated_embedding_dim": concat_data.get("concatenated_embedding_dim"),
                "n_subjects": concat_data.get("n_subjects"),
            })

        self.results["weargait"] = result
        return result

    def _train_rf_baseline(self, force: bool = False) -> Dict:
        task_output_dir = self.output_dir / "Random_Forest_Baseline"
        if (task_output_dir / "summary.json").exists() and not force:
            print("\n✓ RF results found, skipping training.")
            with open(task_output_dir / "summary.json") as f:
                return json.load(f)

        cmd = [sys.executable, str(self.src_dir / "train_gait_rf.py")]
        success = self.run_command(cmd, "TASK 3: Random Forest Baseline")

        result = {
            "task": "rf_baseline",
            "script": "train_gait_rf.py",
            "status": "completed" if success else "failed",
        }
        if success:
            summary_file = task_output_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    result.update(json.load(f))

        self.results["rf"] = result
        return result

    def train_all(self, tasks: List[str], force: bool = False) -> Dict:
        if "all" in tasks:
            tasks = ["pdfe", "weargait", "rf"]

        results = {}
        for task_name in tasks:
            if task_name not in TASK_CONFIGS:
                print(f"⚠ Skipping unknown task: {task_name}")
                continue
            results[task_name] = self.train_task(task_name, force=force)

        self._save_ensemble_summary(results)
        return results

    def _save_ensemble_summary(self, results: Dict):
        summary_file = self.output_dir / "ensemble_summary.json"
        summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "tasks": results,
            "next_step": "Run ensemble_fusion.py to combine prediction artifacts where labels align",
            "status": "ready_for_fusion" if all(r.get("status") == "completed" for r in results.values()) else "some_tasks_failed",
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n✓ Ensemble summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Gait Ensemble Orchestrator - Train 3 independent gait models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            "Comma-separated task names to train:\n"
            "  pdfe       - PDFE severity classification (TCN)\n"
            "  weargait   - per-task WearGait encoders + concatenated embeddings\n"
            "  rf         - Random Forest baseline\n"
            "  all        - Train all three tasks (default)\n"
            "Example: --tasks pdfe,weargait"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for PyTorch models: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain tasks even if results already exist",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    tasks = ["pdfe", "weargait", "rf"] if args.tasks.lower() == "all" else [t.strip() for t in args.tasks.split(",")]
    project_root = Path(__file__).resolve().parents[3]

    print("\n" + "=" * 70)
    print("GAIT ENSEMBLE ORCHESTRATOR")
    print("=" * 70)
    print(f"\nTasks to train: {', '.join(tasks)}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")

    orchestrator = EnsembleOrchestrator(project_root=project_root, device=args.device, seed=args.seed)
    results = orchestrator.train_all(tasks, force=args.force)

    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING SUMMARY")
    print("=" * 70)
    for task_name, result in results.items():
        status = result.get("status", "unknown")
        print(f"\n{task_name.upper():15} -> {status:15}", end="")

    if all(r.get("status") == "completed" for r in results.values()):
        print("\n\n✓ All tasks completed successfully!")
        print("\nNext step: Fuse predictions with")
        print(f"  python {orchestrator.src_dir / 'ensemble_fusion.py'} --strategy all")
    else:
        print("\n\n✗ Some tasks failed. Check output above for details.")

    print(f"\nResults saved to: {orchestrator.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
