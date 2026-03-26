from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimodal_fusion.fusion import LateFusionModel
from multimodal_fusion.loaders import load_handwriting_predictions, load_speech_predictions


GAIT_REPRESENTATIONS = {
    'TUG': PROJECT_ROOT / 'outputs' / 'unimodal_gait' / 'weargait_dl_embeddings' / 'TUG' / 'weargait_subject_embeddings.npz',
    'Concat': PROJECT_ROOT / 'outputs' / 'unimodal_gait' / 'weargait_concat_embeddings' / 'weargait_concat_subject_embeddings.npz',
}


def build_defaults() -> tuple[Path, Path]:
    outputs_root = PROJECT_ROOT / 'outputs' / 'unimodal_gait'
    default_run_name = f"weargait_multimodal_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = outputs_root / 'runs' / default_run_name
    return outputs_root, run_dir


def parse_args() -> argparse.Namespace:
    outputs_root, run_dir = build_defaults()
    parser = argparse.ArgumentParser(
        description='Compare TUG vs concatenated WearGait as the gait input in downstream multimodal late fusion.'
    )
    parser.add_argument('--outputs-root', type=Path, default=outputs_root)
    parser.add_argument('--run-dir', type=Path, default=run_dir)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--handwriting-model', type=str, default='best', choices=['svm', 'best'])
    parser.add_argument('--speech-model', type=str, default='mean', choices=['catboost', 'cnn', 'mean'])
    return parser.parse_args()


def ensure_fresh_run_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'Run directory already exists: {path}')
    path.mkdir(parents=True, exist_ok=False)


def compute_binary_metrics(y_true: np.ndarray, p1: np.ndarray) -> dict[str, float]:
    y_pred = (p1 >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, p1)),
        'ap': float(average_precision_score(y_true, p1)),
        'subjects': int(len(y_true)),
    }


def load_gait_embeddings(npz_path: Path) -> dict[str, np.ndarray]:
    arr = np.load(npz_path, allow_pickle=True)
    return {
        'subject_ids': arr['subject_ids'].astype(str),
        'y': arr['y'].astype(int),
        'X': arr['X_emb'].astype(np.float32),
    }


def build_gait_oof_predictions(npz_path: Path, n_splits: int, n_repeats: int, seed: int) -> tuple[dict, pd.DataFrame]:
    data = load_gait_embeddings(npz_path)
    X = data['X']
    y = data['y']
    subject_ids = data['subject_ids']

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    prob_sum = np.zeros(len(y), dtype=np.float64)
    pred_count = np.zeros(len(y), dtype=np.int32)
    split_rows: list[dict] = []

    for split_id, (tr, va) in enumerate(splitter.split(X, y), start=1):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')),
        ])
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[va])[:, 1]
        pred = (proba >= 0.5).astype(int)
        prob_sum[va] += proba
        pred_count[va] += 1
        split_rows.append({
            'split': split_id,
            'accuracy': accuracy_score(y[va], pred),
            'f1': f1_score(y[va], pred, zero_division=0),
            'auc': roc_auc_score(y[va], proba) if len(np.unique(y[va])) > 1 else np.nan,
            'n_validation': int(len(va)),
        })

    if np.any(pred_count == 0):
        missing = int((pred_count == 0).sum())
        raise RuntimeError(f'{missing} subjects received no OOF prediction.')

    p1 = prob_sum / pred_count
    y_proba = np.column_stack([1.0 - p1, p1])
    modality = {
        'subject_ids': subject_ids,
        'y_true': y,
        'y_proba': y_proba,
        'task': 'diagnosis',
        'note': (
            'WearGait embedding converted to subject-level OOF probabilities using '
            f'RepeatedStratifiedKFold logistic regression ({n_splits} folds x {n_repeats} repeats).'
        ),
    }
    split_df = pd.DataFrame(split_rows)
    split_df['prediction_count_per_subject'] = int(pred_count[0])
    return modality, split_df


def to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serialisable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def make_strategy_plot(plot_df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style='whitegrid', context='talk')
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x='strategy', y='auc_mean', hue='gait_representation', palette='Set2')
    plt.ylim(0.0, 1.0)
    plt.title('Downstream Multimodal Comparison: TUG vs Concat Gait Input')
    plt.xlabel('Fusion Strategy')
    plt.ylabel('Bootstrap-Simulated Fusion AUC')
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.annotate(f'{h:.3f}', (patch.get_x() + patch.get_width() / 2, h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_weight_plot(weight_df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style='whitegrid', context='talk')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=weight_df, x='modality', y='weight', hue='gait_representation', palette='Set3')
    plt.ylim(0.0, 1.0)
    plt.title('AUC-Weighted Late Fusion Weights by Gait Input')
    plt.xlabel('Modality')
    plt.ylabel('Static Weight')
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.annotate(f'{h:.3f}', (patch.get_x() + patch.get_width() / 2, h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_notes(run_dir: Path, summary_df: pd.DataFrame, modality_df: pd.DataFrame, args: argparse.Namespace) -> None:
    best = summary_df.sort_values('auc_mean', ascending=False).iloc[0]
    gait_rank = modality_df[modality_df['modality'] == 'gait'].sort_values('auc', ascending=False)
    text = f'''# WearGait Downstream Multimodal Comparison

Run directory: `{run_dir}`

## What this run does
- Builds subject-level gait probabilities from the current `TUG` and `Concat` embeddings
- Uses repeated stratified CV logistic regression on the gait embeddings
- Compares downstream multimodal late fusion using the same handwriting and speech inputs
- Evaluates four fusion strategies from `multimodal_fusion.fusion.LateFusionModel`

## Inputs used
- Handwriting model: `{args.handwriting_model}`
- Speech model: `{args.speech_model}`
- Gait candidates: `TUG`, `Concat`

## Important caveat
- These multimodal fusion results are simulation-based because the modality cohorts are not the same subjects.
- The fusion AUCs are useful for relative comparison between `TUG` and `Concat`, but they are not a substitute for a true aligned multimodal cohort evaluation.

## Key takeaways
- Best downstream fusion configuration by simulated AUC: `{best['gait_representation']} + {best['strategy']}`
- Best gait-only input by unimodal AUC in this run: `{gait_rank.iloc[0]['gait_representation']}` ({gait_rank.iloc[0]['auc']:.3f})
- The more promising gait branch for the next multimodal phase is whichever stays stronger across both unimodal and fusion views.

## Fusion strategy ranking
{summary_df[['gait_representation', 'strategy', 'auc_mean', 'auc_ci_lo', 'auc_ci_hi']].to_string(index=False)}

## Modality metrics
{modality_df[['gait_representation', 'modality', 'auc', 'f1', 'accuracy', 'weight']].to_string(index=False)}

## Recommended next step
- Carry the stronger of `TUG` vs `Concat` forward as the primary gait input for multimodal fusion experiments.
- Keep the other as an ablation so the presentation can show that the choice was tested rather than assumed.
'''
    (run_dir / 'notes.md').write_text(text, encoding='utf-8')


def main() -> None:
    args = parse_args()
    ensure_fresh_run_dir(args.run_dir)

    handwriting = load_handwriting_predictions(model=args.handwriting_model)
    speech = load_speech_predictions(speech_model=args.speech_model)

    gait_split_frames = []
    gait_oof_frames = []
    modality_rows = []
    fusion_rows = []
    detailed_results = {}

    for gait_name, npz_path in GAIT_REPRESENTATIONS.items():
        gait_modality, split_df = build_gait_oof_predictions(npz_path, args.n_splits, args.n_repeats, args.seed)
        gait_modality['note'] = gait_modality['note'] + f' Representation={gait_name}.'
        modality_data = {
            'gait': gait_modality,
            'handwriting': handwriting,
            'speech': speech,
        }

        model = LateFusionModel(strategy='auc_weighted').fit(modality_data)
        all_results = model.evaluate_all_strategies(modality_data)
        detailed_results[gait_name] = to_serialisable(all_results)

        gait_split = split_df.copy()
        gait_split['gait_representation'] = gait_name
        gait_split_frames.append(gait_split)

        gait_p1 = gait_modality['y_proba'][:, 1]
        gait_oof = pd.DataFrame({
            'gait_representation': gait_name,
            'subject_id': gait_modality['subject_ids'],
            'y_true': gait_modality['y_true'],
            'p_pd': gait_p1,
        })
        gait_oof_frames.append(gait_oof)

        for modality_name, metrics in model.evaluate_unimodal().items():
            modality_rows.append({
                'gait_representation': gait_name,
                'modality': modality_name,
                'subjects': metrics['n_subjects'],
                'prevalence': metrics['prevalence'],
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'ap': metrics['ap'],
                'weight': model.weights_[modality_name],
            })

        for strategy_key, metrics in all_results.items():
            if not strategy_key.startswith('fusion_'):
                continue
            fusion_rows.append({
                'gait_representation': gait_name,
                'strategy': strategy_key.replace('fusion_', ''),
                'auc_mean': metrics['auc_mean'],
                'auc_std': metrics['auc_std'],
                'auc_ci_lo': metrics['auc_ci_lo'],
                'auc_ci_hi': metrics['auc_ci_hi'],
                'n_bootstrap': metrics['n_bootstrap'],
            })

    gait_split_df = pd.concat(gait_split_frames, ignore_index=True)
    gait_oof_df = pd.concat(gait_oof_frames, ignore_index=True)
    modality_df = pd.DataFrame(modality_rows)
    fusion_df = pd.DataFrame(fusion_rows).sort_values(['auc_mean', 'gait_representation'], ascending=[False, True]).reset_index(drop=True)

    gait_split_df.to_csv(args.run_dir / 'gait_repeated_cv_split_metrics.csv', index=False)
    gait_oof_df.to_csv(args.run_dir / 'gait_oof_probabilities.csv', index=False)
    modality_df.to_csv(args.run_dir / 'modality_metrics.csv', index=False)
    fusion_df.to_csv(args.run_dir / 'fusion_strategy_summary.csv', index=False)
    (args.run_dir / 'fusion_detailed_results.json').write_text(json.dumps(detailed_results, indent=2), encoding='utf-8')

    make_strategy_plot(fusion_df, args.run_dir / 'fusion_strategy_comparison.png')
    make_weight_plot(modality_df[['gait_representation', 'modality', 'weight']].copy(), args.run_dir / 'modality_weight_comparison.png')
    write_notes(args.run_dir, fusion_df, modality_df, args)

    run_summary = {
        'run_dir': str(args.run_dir),
        'handwriting_model': args.handwriting_model,
        'speech_model': args.speech_model,
        'n_splits': args.n_splits,
        'n_repeats': args.n_repeats,
        'best_fusion_by_auc': fusion_df.iloc[0].to_dict(),
        'best_gait_unimodal_auc': modality_df[modality_df['modality'] == 'gait'].sort_values('auc', ascending=False).iloc[0].to_dict(),
        'outputs': {
            'gait_repeated_cv_split_metrics_csv': str(args.run_dir / 'gait_repeated_cv_split_metrics.csv'),
            'gait_oof_probabilities_csv': str(args.run_dir / 'gait_oof_probabilities.csv'),
            'modality_metrics_csv': str(args.run_dir / 'modality_metrics.csv'),
            'fusion_strategy_summary_csv': str(args.run_dir / 'fusion_strategy_summary.csv'),
            'fusion_detailed_results_json': str(args.run_dir / 'fusion_detailed_results.json'),
            'fusion_strategy_comparison_png': str(args.run_dir / 'fusion_strategy_comparison.png'),
            'modality_weight_comparison_png': str(args.run_dir / 'modality_weight_comparison.png'),
            'notes_md': str(args.run_dir / 'notes.md'),
        },
    }
    (args.run_dir / 'run_summary.json').write_text(json.dumps(to_serialisable(run_summary), indent=2), encoding='utf-8')

    print(json.dumps(to_serialisable(run_summary), indent=2))
    print('\nFusion strategy summary:')
    print(fusion_df.to_string(index=False))
    print('\nModality metrics:')
    print(modality_df.to_string(index=False))


if __name__ == '__main__':
    main()
