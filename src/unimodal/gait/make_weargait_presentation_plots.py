from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TASKS = ['SelfPace', 'HurriedPace', 'TUG']


def load_task_data(root: Path):
    rows = []
    fold_rows = []
    for task in TASKS:
        summary = json.loads((root / 'weargait_dl_embeddings' / task / 'summary.json').read_text())
        metrics = pd.read_csv(root / 'weargait_dl_embeddings' / task / 'cv_metrics.csv')
        rows.append({
            'task': task,
            'subjects': summary['n_subjects'],
            'windows': summary['n_windows'],
            'channels': summary['n_channels'],
            'embedding_dim': summary['embedding_dim'],
            'mean_acc': summary['mean_acc'],
            'mean_f1': summary['mean_f1'],
            'mean_auc': summary['mean_auc'],
        })
        tmp = metrics.copy()
        tmp['task'] = task
        fold_rows.append(tmp)
    return pd.DataFrame(rows), pd.concat(fold_rows, ignore_index=True)


def plot_task_metrics(task_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = task_df.melt(
        id_vars=['task'],
        value_vars=['mean_acc', 'mean_f1', 'mean_auc'],
        var_name='metric',
        value_name='value',
    )
    metric_labels = {'mean_acc': 'Accuracy', 'mean_f1': 'F1', 'mean_auc': 'AUC'}
    plot_df['metric'] = plot_df['metric'].map(metric_labels)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='task', y='value', hue='metric', palette='Set2')
    plt.ylim(0.0, 1.0)
    plt.title('WearGait Separate-Task TCN Performance')
    plt.ylabel('Score')
    plt.xlabel('Task')
    for i, patch in enumerate(plt.gca().patches):
        height = patch.get_height()
        plt.gca().annotate(f'{height:.3f}', (patch.get_x() + patch.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_fold_heatmap(fold_df: pd.DataFrame, out_path: Path) -> None:
    auc_heat = fold_df.pivot(index='task', columns='fold', values='auc').loc[TASKS]
    f1_heat = fold_df.pivot(index='task', columns='fold', values='f1').loc[TASKS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(auc_heat, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0.4, vmax=0.95, ax=axes[0])
    axes[0].set_title('Per-Fold AUC')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Task')

    sns.heatmap(f1_heat, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0.6, vmax=0.9, ax=axes[1])
    axes[1].set_title('Per-Fold F1')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('')

    plt.suptitle('WearGait Cross-Validation Stability by Task', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def plot_coverage(task_df: pd.DataFrame, audit_summary: dict, concat_summary: dict, out_path: Path) -> None:
    stages = pd.DataFrame([
        {'stage': 'Indexed cohort', 'subjects': 185},
        {'stage': 'Old channel set\n(best task)', 'subjects': max(audit_summary['usable_subjects_by_task'].values())},
        {'stage': 'IMU-only\nSelfPace', 'subjects': int(task_df.loc[task_df.task == 'SelfPace', 'subjects'].iloc[0])},
        {'stage': 'IMU-only\nHurriedPace', 'subjects': int(task_df.loc[task_df.task == 'HurriedPace', 'subjects'].iloc[0])},
        {'stage': 'IMU-only\nTUG', 'subjects': int(task_df.loc[task_df.task == 'TUG', 'subjects'].iloc[0])},
        {'stage': 'Concatenated\nintersection', 'subjects': concat_summary['n_subjects']},
    ])

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=stages, x='stage', y='subjects', palette='crest')
    plt.title('Coverage Recovery and Fusion-Ready Cohort Size')
    plt.ylabel('Subjects')
    plt.xlabel('Pipeline stage')
    plt.ylim(0, 200)
    for patch in ax.patches:
        h = patch.get_height()
        ax.annotate(f'{int(h)}', (patch.get_x() + patch.get_width() / 2, h),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_fusion_ready(task_df: pd.DataFrame, concat_summary: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    subjects_df = task_df[['task', 'subjects']].copy()
    sns.barplot(data=subjects_df, x='task', y='subjects', palette='Set1', ax=axes[0])
    axes[0].set_title('Subjects Retained Per Task Encoder')
    axes[0].set_ylim(0, 200)
    for patch in axes[0].patches:
        h = patch.get_height()
        axes[0].annotate(f'{int(h)}', (patch.get_x() + patch.get_width() / 2, h),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

    dims_df = pd.DataFrame([
        {'representation': 'Per-task embedding', 'dim': 256},
        {'representation': 'Concatenated gait embedding', 'dim': concat_summary['concatenated_embedding_dim']},
    ])
    sns.barplot(data=dims_df, x='representation', y='dim', palette='flare', ax=axes[1])
    axes[1].set_title('Fusion-Ready Representation Size')
    axes[1].set_ylim(0, 850)
    for patch in axes[1].patches:
        h = patch.get_height()
        axes[1].annotate(f'{int(h)}', (patch.get_x() + patch.get_width() / 2, h),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

    plt.suptitle('Task-Aware WearGait Representation for Downstream Fusion', y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[3] / 'outputs' / 'unimodal_gait'
    out_dir = root / 'presentation'
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style='whitegrid', context='talk')

    task_df, fold_df = load_task_data(root)
    audit_summary = json.loads((root / 'weargait_audit' / 'summary.json').read_text())
    concat_summary = json.loads((root / 'weargait_concat_embeddings' / 'summary.json').read_text())

    plot_task_metrics(task_df, out_dir / 'weargait_task_metrics_bar.png')
    plot_fold_heatmap(fold_df, out_dir / 'weargait_fold_heatmaps.png')
    plot_coverage(task_df, audit_summary, concat_summary, out_dir / 'weargait_coverage_recovery.png')
    plot_fusion_ready(task_df, concat_summary, out_dir / 'weargait_fusion_ready_summary.png')

    summary = {
        'plots': [
            'weargait_task_metrics_bar.png',
            'weargait_fold_heatmaps.png',
            'weargait_coverage_recovery.png',
            'weargait_fusion_ready_summary.png',
        ],
        'task_metrics': task_df.to_dict(orient='records'),
        'concat_subjects': concat_summary['n_subjects'],
        'concat_dim': concat_summary['concatenated_embedding_dim'],
        'note': 'Fusion plot here refers to the fusion-ready concatenated gait embedding, not the older gait fusion_results artifact.'
    }
    (out_dir / 'plot_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
