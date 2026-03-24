"""
Ensemble Fusion Strategy
=======================

Combines outputs from 3 independent gait tasks:
1. PDFE severity predictions
2. WearGait embeddings / predictions
3. RF baseline predictions

Tests multiple fusion strategies:
- Weighted averaging
- Stacking (meta-learner)
- Late fusion with calibration
- Voting ensemble

Usage:
    python ensemble_fusion.py --strategy weighted
    python ensemble_fusion.py --strategy stacking --cv 5
    python ensemble_fusion.py --strategy all --save-models
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FusionConfig:
    """Configuration for fusion strategy."""
    name: str
    method: str  # 'weighted', 'stacking', 'voting', 'calibrated'
    n_jobs: int = -1
    cv_folds: int = 5


# Available fusion strategies
FUSION_STRATEGIES = {
    'weighted': FusionConfig(
        name='Weighted_Average_Fusion',
        method='weighted',
        cv_folds=5,
    ),
    'stacking': FusionConfig(
        name='Stacking_Meta_Learner',
        method='stacking',
        cv_folds=5,
    ),
    'voting': FusionConfig(
        name='Voting_Ensemble',
        method='voting',
        cv_folds=5,
    ),
    'calibrated': FusionConfig(
        name='Calibrated_Late_Fusion',
        method='calibrated',
        cv_folds=5,
    ),
}

TASK_OUTPUT_DIRS = {
    'pdfe': 'PDFE_Severity_Classification',
    'weargait': 'WearGait_Task_Embeddings',
    'rf': 'Random_Forest_Baseline',
}


class EnsembleFusion:
    """Fuses predictions from multiple gait tasks."""
    
    def __init__(
        self,
        project_root: Path,
        seed: int = 42,
    ):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / 'outputs' / 'unimodal_gait'
        self.ensemble_dir = self.output_dir / 'fusion_results'
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        self._seed_everything()
        
        print(f"✓ Initialized EnsembleFusion")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Fusion dir: {self.ensemble_dir}")
    
    def _seed_everything(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
    
    def load_task_predictions(self) -> Dict[str, np.ndarray]:
        """
        Load predictions from each trained task.
        
        Expected structure:
            outputs/unimodal_gait/
            ├── PDFE_Severity_Classification/
            │   └── predictions.npz
            ├── WearGait_Task_Embeddings/
            │   └── predictions.npz
            └── Random_Forest_Baseline/
                └── predictions.npz
        
        Returns:
            Dict mapping task name -> (y_true, y_pred, y_proba)
        """
        print("\nLoading task predictions...")
        
        predictions = {}
        
        # Load explicit task directories to avoid missing RF/WearGait outputs.
        for task_key, task_folder in TASK_OUTPUT_DIRS.items():
            task_dir = self.output_dir / task_folder
            pred_file = task_dir / 'predictions.npz'
            if not pred_file.exists():
                print(f"  - Missing predictions for {task_key}: {pred_file}")
                continue

            data = np.load(pred_file)
            if 'y_true' not in data or 'y_pred' not in data:
                print(f"  - Invalid predictions schema for {task_key}: {pred_file}")
                continue

            y_true = np.array(data['y_true'])
            y_pred = np.array(data['y_pred'])
            y_proba = np.array(data['y_proba']) if 'y_proba' in data else None
            if y_proba is not None and y_proba.ndim == 1:
                y_proba = np.column_stack([1.0 - y_proba, y_proba])

            if 'subject_ids' in data:
                subject_ids = np.array(data['subject_ids']).astype(str)
            else:
                # Fallback when subject identifiers are unavailable.
                subject_ids = np.array([str(i) for i in range(len(y_true))], dtype=str)

            predictions[task_key] = {
                'task_dir': task_folder,
                'subject_ids': subject_ids,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba,
            }
            print(f"  ✓ Loaded {task_key} ({len(subject_ids)} rows)")
        
        if not predictions:
            print("  ⚠ No task predictions found. Run gait_ensemble_orchestrator.py first.")
        else:
            predictions = self._align_predictions_by_subject(predictions)
        
        return predictions

    def _align_predictions_by_subject(self, predictions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Align task predictions to the intersection of subject IDs.

        Fusion only makes sense when predictions correspond to the same subjects.
        """
        if not predictions:
            return predictions

        subject_sets = [set(v['subject_ids'].tolist()) for v in predictions.values()]
        common_subjects = sorted(set.intersection(*subject_sets)) if subject_sets else []

        if not common_subjects:
            print("  ⚠ No common subject IDs across tasks; fusion cannot align predictions.")
            return {}

        aligned = {}
        for task_name, payload in predictions.items():
            sid_to_idx = {sid: i for i, sid in enumerate(payload['subject_ids'].tolist())}
            idx = [sid_to_idx[sid] for sid in common_subjects]

            task_aligned = {
                'task_dir': payload['task_dir'],
                'subject_ids': np.array(common_subjects, dtype=str),
                'y_true': np.array(payload['y_true'])[idx],
                'y_pred': np.array(payload['y_pred'])[idx],
                'y_proba': np.array(payload['y_proba'])[idx] if payload['y_proba'] is not None else None,
            }
            aligned[task_name] = task_aligned

        # Use first task as reference labels and report mismatches.
        ref_task = next(iter(aligned.keys()))
        ref_y = aligned[ref_task]['y_true']
        for task_name, payload in aligned.items():
            mismatches = int(np.sum(payload['y_true'] != ref_y))
            if mismatches > 0:
                print(f"  ⚠ Label mismatch vs {ref_task} in {task_name}: {mismatches} subjects")
        print(f"  ✓ Aligned {len(common_subjects)} common subjects across {len(aligned)} tasks")

        return aligned
    
    def fuse_weighted(
        self,
        predictions: Dict,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simple weighted average of probability predictions.
        
        Args:
            predictions: Task predictions dict
            weights: Optional task weights (will learn if None)
            
        Returns:
            (fused_proba, results_dict)
        """
        print("\n--- Weighted Average Fusion ---")
        
        if not predictions:
            print("No predictions to fuse.")
            return None, {}
        
        # If weights not provided, learn them from validation data
        # For now, use equal weights
        if weights is None:
            weights = {task: 1.0 / len(predictions) for task in predictions}
        
        # Get first task's y_true
        y_true = list(predictions.values())[0]['y_true']
        
        # Stack probability predictions
        proba_stack = []
        for task_name, pred in predictions.items():
            if pred['y_proba'] is not None:
                p = pred['y_proba']
                w = weights.get(task_name, 1.0 / len(predictions))
                proba_stack.append(p * w)
                print(f"  Task: {task_name}, weight: {w:.3f}")
        
        # Weighted sum
        fused_proba = np.sum(proba_stack, axis=0)
        fused_preds = (fused_proba[:, 1] > 0.5).astype(int)
        
        # Evaluate
        results = {
            'method': 'weighted_average',
            'accuracy': accuracy_score(y_true, fused_preds),
            'f1': f1_score(y_true, fused_preds, zero_division=0),
            'roc_auc': roc_auc_score(y_true, fused_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.0,
            'weights': weights,
        }
        
        print(f"\n  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
        
        return fused_proba, results
    
    def fuse_stacking(
        self,
        predictions: Dict,
        meta_model = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Stacking: Use predictions from all tasks as features for meta-learner.
        
        Args:
            predictions: Task predictions dict
            meta_model: Scikit-learn classifier for meta-learner (default: LogisticRegression)
            
        Returns:
            (meta_proba, results_dict)
        """
        print("\n--- Stacking Fusion ---")
        
        if not predictions:
            return None, {}
        
        if meta_model is None:
            meta_model = LogisticRegression(random_state=self.seed, max_iter=1000)
        
        y_true = list(predictions.values())[0]['y_true']
        
        # Stack probability predictions as features for meta-learner
        X_meta = []
        for task_name, pred in predictions.items():
            if pred['y_proba'] is not None:
                X_meta.append(pred['y_proba'])
            else:
                # Fallback: use binary predictions as one-hot
                y = pred['y_pred']
                X_meta.append(np.column_stack([1 - y, y]).astype(float))
            print(f"  Input from {task_name}: shape {X_meta[-1].shape}")
        
        X_meta = np.hstack(X_meta)
        print(f"  Meta-learner input: {X_meta.shape}")
        
        # Train meta-learner
        meta_model.fit(X_meta, y_true)
        meta_proba = meta_model.predict_proba(X_meta)
        meta_preds = meta_model.predict(X_meta)
        
        results = {
            'method': 'stacking',
            'meta_model_type': type(meta_model).__name__,
            'accuracy': accuracy_score(y_true, meta_preds),
            'f1': f1_score(y_true, meta_preds, zero_division=0),
            'roc_auc': roc_auc_score(y_true, meta_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.0,
        }
        
        print(f"\n  Meta-learner: {type(meta_model).__name__}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
        
        return meta_proba, results
    
    def fuse_voting(
        self,
        predictions: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Hard/soft voting ensemble.
        """
        print("\n--- Voting Ensemble ---")
        
        if not predictions:
            return None, {}
        
        y_true = list(predictions.values())[0]['y_true']
        
        # Voting: majority of prediction outcomes
        voting_counts = np.zeros_like(y_true, dtype=float)
        for task_name, pred in predictions.items():
            voting_counts += pred['y_pred']
            print(f"  Input from {task_name}")
        
        # Normalize: voting score = fraction of tasks predicting class 1
        voting_score = voting_counts / len(predictions)
        voting_preds = (voting_score > 0.5).astype(int)
        
        results = {
            'method': 'voting',
            'n_tasks': len(predictions),
            'accuracy': accuracy_score(y_true, voting_preds),
            'f1': f1_score(y_true, voting_preds, zero_division=0),
        }
        
        print(f"\n  Number of tasks: {results['n_tasks']}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        
        return voting_score[:, np.newaxis], results
    
    def evaluate_fusion(self, strategy: str = 'all') -> Dict:
        """
        Evaluate multiple fusion strategies and compare.
        
        Args:
            strategy: 'weighted', 'stacking', 'voting', 'calibrated', or 'all'
            
        Returns:
            Dict with results from all evaluated strategies
        """
        print("\n" + "="*60)
        print("FUSION STRATEGY EVALUATION")
        print("="*60)
        
        # Load predictions
        predictions = self.load_task_predictions()
        
        if not predictions:
            print("\nNo task predictions found. Cannot proceed with fusion.")
            return {}
        
        # Run fusion strategies
        fusion_results = {}
        
        if strategy in ['weighted', 'all']:
            _, weighted_res = self.fuse_weighted(predictions)
            fusion_results['weighted'] = weighted_res
        
        if strategy in ['stacking', 'all']:
            _, stacking_res = self.fuse_stacking(predictions)
            fusion_results['stacking'] = stacking_res
        
        if strategy in ['voting', 'all']:
            _, voting_res = self.fuse_voting(predictions)
            fusion_results['voting'] = voting_res
        
        # Save comparison
        self._save_fusion_summary(fusion_results)
        
        return fusion_results
    
    def _save_fusion_summary(self, results: Dict):
        """Save fusion results summary."""
        summary_file = self.ensemble_dir / 'fusion_comparison.json'
        
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'strategies': results,
            'best_strategy': max(
                results.items(),
                key=lambda x: x[1].get('accuracy', 0) + x[1].get('f1', 0)
            )[0] if results else None,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✓ Fusion summary saved: {summary_file}")
    
    def create_comparison_plot(self, results: Dict):
        """Create visualization comparing fusion strategies."""
        if not results:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        strategies = list(results.keys())
        accuracies = [results[s].get('accuracy', 0) for s in strategies]
        f1_scores = [results[s].get('f1', 0) for s in strategies]
        aucs = [results[s].get('roc_auc', 0) for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.8
        
        axes[0].bar(x, accuracies, width, color='steelblue')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, rotation=45, ha='right')
        axes[0].set_ylim([0, 1])
        
        axes[1].bar(x, f1_scores, width, color='darkorange')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('F1 Score Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strategies, rotation=45, ha='right')
        axes[1].set_ylim([0, 1])
        
        axes[2].bar(x, aucs, width, color='seagreen')
        axes[2].set_ylabel('ROC AUC')
        axes[2].set_title('ROC AUC Comparison')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(strategies, rotation=45, ha='right')
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plot_file = self.ensemble_dir / 'fusion_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plot saved: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Ensemble Fusion')
    parser.add_argument(
        '--strategy',
        choices=['weighted', 'stacking', 'voting', 'all'],
        default='all',
        help='Fusion strategy to evaluate',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-models', action='store_true')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[3]
    fusion = EnsembleFusion(project_root=project_root, seed=args.seed)
    
    results = fusion.evaluate_fusion(strategy=args.strategy)
    
    if results:
        fusion.create_comparison_plot(results)
    
    print("\n" + "="*60)
    print("FUSION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {fusion.ensemble_dir}")


if __name__ == '__main__':
    main()
