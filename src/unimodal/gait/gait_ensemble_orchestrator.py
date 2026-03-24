"""
Gait Ensemble Orchestrator
========================

Unified training pipeline for 3 independent gait tasks:
1. Task 1: PDFE Severity Classification (figshare) - calls train_gait.py
2. Task 2: Wearable Task Embeddings (SelfPace, HurriedPace, TUG) - calls train_weargait_embeddings.py
3. Task 3: Random Forest Baseline (engineered features) - calls train_gait_rf.py

Each task is trained independently with proper stratified cross-validation.
Results and predictions are saved separately for later fusion.

Usage:
    python gait_ensemble_orchestrator.py --tasks all
    python gait_ensemble_orchestrator.py --tasks pdfe,weargait
    python gait_ensemble_orchestrator.py --tasks rf --device cpu
    
    # Then fuse:
    python ensemble_fusion.py --strategy all
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ==================== CONFIG ====================
# Task names and their associated scripts
TASK_CONFIGS = {
    'pdfe': {
        'name': 'PDFE_Severity_Classification',
        'script': 'train_gait.py',
        'description': 'Temporal Convolutional Network on figshare IMU data',
    },
    'weargait': {
        'name': 'WearGait_Task_Embeddings',
        'script': 'train_weargait_embeddings.py',
        'description': 'Temporal Convolutional Network on wearable sensor data',
    },
    'rf': {
        'name': 'Random_Forest_Baseline',
        'script': 'train_gait_rf.py',
        'description': 'Random Forest with engineered gait features',
    },
}


class EnsembleOrchestrator:
    """
    Orchestrator for multi-task gait training.
    
    Wraps existing standalone scripts (train_gait.py, train_weargait_embeddings.py, etc.)
    and coordinates their outputs for fusion.
    """
    
    def __init__(
        self,
        project_root: Path,
        device: str = 'cuda',
        seed: int = 42,
    ):
        self.project_root = Path(project_root)
        self.device = device
        self.seed = seed
        
        # Paths
        self.src_dir = self.project_root / 'src' / 'unimodal' / 'gait'
        self.data_dir = self.project_root / 'data' / 'gait'
        self.output_dir = self.project_root / 'outputs' / 'unimodal_gait'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        
        print(f"✓ Initialized EnsembleOrchestrator")
        print(f"  Script dir: {self.src_dir}")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Output dir: {self.output_dir}")
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """
        Run a shell command and report success/failure.
        
        Args:
            cmd: Command as list (for subprocess)
            description: Human-readable description of what's running
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True, text=True)
            print(f"\n✓ {description} COMPLETED\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {description} FAILED")
            print(f"Error: {e}\n")
            return False
    
    def train_task(
        self,
        task_name: str,
        force: bool = False,
    ) -> Dict:
        """
        Train a single task by calling its standalone script.
        
        Args:
            task_name: 'pdfe', 'weargait', or 'rf'
            force: Rerun even if results exist
            
        Returns:
            Dict with task results summary
        """
        if task_name == 'pdfe':
            return self._train_pdfe_severity(force)
        elif task_name == 'weargait':
            return self._train_weargait_embeddings(force)
        elif task_name == 'rf':
            return self._train_rf_baseline(force)
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def _train_pdfe_severity(self, force: bool = False) -> Dict:
        """Train PDFE severity classification using train_gait.py."""
        task_output_dir = self.output_dir / 'PDFE_Severity_Classification'
        
        # Check if already done
        if (task_output_dir / 'summary.json').exists() and not force:
            print(f"\n✓ PDFE results found, skipping training.")
            with open(task_output_dir / 'summary.json') as f:
                return json.load(f)
        
        # Run training script
        cmd = [sys.executable, str(self.src_dir / 'train_gait.py')]
        success = self.run_command(cmd, "TASK 1: PDFE Severity Classification (TCN)")
        
        result = {
            'task': 'pdfe_severity',
            'script': 'train_gait.py',
            'status': 'completed' if success else 'failed',
        }
        
        if success:
            # Try to load summary
            summary_file = task_output_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    result.update(json.load(f))
        
        self.results['pdfe'] = result
        return result
    
    def _train_weargait_embeddings(self, force: bool = False) -> Dict:
        """Train WearGait embeddings using train_weargait_embeddings.py."""
        task_output_dir = self.output_dir / 'WearGait_Task_Embeddings'
        
        # Check if already done
        if (task_output_dir / 'summary.json').exists() and not force:
            print(f"\n✓ WearGait results found, skipping training.")
            with open(task_output_dir / 'summary.json') as f:
                return json.load(f)
        
        # Run training script
        cmd = [
            sys.executable,
            str(self.src_dir / 'train_weargait_embeddings.py'),
            '--output-dir',
            str(task_output_dir),
        ]
        success = self.run_command(cmd, "TASK 2: WearGait Task Embeddings (TCN)")
        
        result = {
            'task': 'weargait_embeddings',
            'script': 'train_weargait_embeddings.py',
            'status': 'completed' if success else 'failed',
        }
        
        if success:
            summary_file = task_output_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    result.update(json.load(f))
        
        self.results['weargait'] = result
        return result
    
    def _train_rf_baseline(self, force: bool = False) -> Dict:
        """Train RF baseline using train_gait_rf.py."""
        task_output_dir = self.output_dir / 'Random_Forest_Baseline'
        
        # Check if already done
        if (task_output_dir / 'summary.json').exists() and not force:
            print(f"\n✓ RF results found, skipping training.")
            with open(task_output_dir / 'summary.json') as f:
                return json.load(f)
        
        # Run training script
        cmd = [sys.executable, str(self.src_dir / 'train_gait_rf.py')]
        success = self.run_command(cmd, "TASK 3: Random Forest Baseline")
        
        result = {
            'task': 'rf_baseline',
            'script': 'train_gait_rf.py',
            'status': 'completed' if success else 'failed',
        }
        
        if success:
            summary_file = task_output_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    result.update(json.load(f))
        
        self.results['rf'] = result
        return result
    
    def train_all(self, tasks: List[str], force: bool = False) -> Dict:
        """
        Train multiple tasks sequentially.
        
        Args:
            tasks: List of task names ('pdfe', 'weargait', 'rf', or ['all'])
            force: Retrain even if results exist
            
        Returns:
            Dict of all task results
        """
        if 'all' in tasks:
            tasks = ['pdfe', 'weargait', 'rf']
        
        results = {}
        for task_name in tasks:
            if task_name not in TASK_CONFIGS:
                print(f"⚠ Skipping unknown task: {task_name}")
                continue
            
            results[task_name] = self.train_task(task_name, force=force)
        
        # Save ensemble summary
        self._save_ensemble_summary(results)
        
        return results
    
    def _save_ensemble_summary(self, results: Dict):
        """Save unified summary with all task results."""
        summary_file = self.output_dir / 'ensemble_summary.json'
        
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'tasks': results,
            'next_step': 'Run ensemble_fusion.py to combine task outputs',
            'status': 'ready_for_fusion' if all(r.get('status') == 'completed' for r in results.values()) else 'some_tasks_failed',
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✓ Ensemble summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Gait Ensemble Orchestrator - Train 3 independent gait models',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help=(
            "Comma-separated task names to train:\n"
            "  pdfe       - PDFE severity classification (TCN)\n"
            "  weargait   - WearGait embeddings (TCN)\n"
            "  rf         - Random Forest baseline\n"
            "  all        - Train all three tasks (default)\n"
            "Example: --tasks pdfe,weargait"
        ),
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help="Device for PyTorch models: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Retrain tasks even if results already exist',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    
    args = parser.parse_args()
    
    # Parse task list
    if args.tasks.lower() == 'all':
        tasks = ['pdfe', 'weargait', 'rf']
    else:
        tasks = [t.strip() for t in args.tasks.split(',')]
    
    # Initialize and run
    project_root = Path(__file__).resolve().parents[3]
    
    print("\n" + "="*70)
    print("GAIT ENSEMBLE ORCHESTRATOR")
    print("="*70)
    print(f"\nTasks to train: {', '.join(tasks)}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    orchestrator = EnsembleOrchestrator(
        project_root=project_root,
        device=args.device,
        seed=args.seed,
    )
    
    results = orchestrator.train_all(tasks, force=args.force)
    
    # Print summary
    print("\n" + "="*70)
    print("ENSEMBLE TRAINING SUMMARY")
    print("="*70)
    
    for task_name, result in results.items():
        status = result.get('status', 'unknown')
        print(f"\n{task_name.upper():15} -> {status:15}", end="")
    
    # Check if ready for fusion
    if all(r.get('status') == 'completed' for r in results.values()):
        print("\n\n✓ All tasks completed successfully!")
        print("\nNext step: Fuse predictions with")
        print(f"  python {orchestrator.src_dir / 'ensemble_fusion.py'} --strategy all")
    else:
        print("\n\n✗ Some tasks failed. Check output above for details.")
    
    print(f"\nResults saved to: {orchestrator.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
