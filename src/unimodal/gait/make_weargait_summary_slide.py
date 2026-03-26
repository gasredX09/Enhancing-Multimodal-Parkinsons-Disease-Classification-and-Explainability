from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TASKS = ['SelfPace', 'HurriedPace', 'TUG']


def main() -> None:
    root = Path(__file__).resolve().parents[3] / 'outputs' / 'unimodal_gait'
    out_dir = root / 'presentation'
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style='whitegrid', context='talk')

    task_rows = []
    for task in TASKS:
        summary = json.loads((root / 'weargait_dl_embeddings' / task / 'summary.json').read_text())
        task_rows.append({
            'task': task,
            'subjects': summary['n_subjects'],
            'mean_acc': summary['mean_acc'],
            'mean_f1': summary['mean_f1'],
            'mean_auc': summary['mean_auc'],
        })
    task_df = pd.DataFrame(task_rows)

    audit_summary = json.loads((root / 'weargait_audit' / 'summary.json').read_text())
    concat_summary = json.loads((root / 'weargait_concat_embeddings' / 'summary.json').read_text())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metric_df = task_df[['task', 'mean_auc']].copy()
    sns.barplot(data=metric_df, x='task', y='mean_auc', hue='task', dodge=False, palette='Set2', legend=False, ax=axes[0])
    axes[0].set_title('Best Single-Task Signal')
    axes[0].set_xlabel('WearGait task')
    axes[0].set_ylabel('AUC')
    axes[0].set_ylim(0.0, 1.0)
    for patch in axes[0].patches:
        h = patch.get_height()
        axes[0].annotate(f'{h:.3f}', (patch.get_x() + patch.get_width() / 2, h),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

    coverage_df = pd.DataFrame([
        {'stage': 'Indexed', 'subjects': 185},
        {'stage': 'Old channels', 'subjects': max(audit_summary['usable_subjects_by_task'].values())},
        {'stage': 'IMU-only concat', 'subjects': concat_summary['n_subjects']},
    ])
    sns.barplot(data=coverage_df, x='stage', y='subjects', hue='stage', dodge=False, palette='crest', legend=False, ax=axes[1])
    axes[1].set_title('Coverage Recovery')
    axes[1].set_xlabel('Pipeline stage')
    axes[1].set_ylabel('Subjects')
    axes[1].set_ylim(0, 200)
    for patch in axes[1].patches:
        h = patch.get_height()
        axes[1].annotate(f'{int(h)}', (patch.get_x() + patch.get_width() / 2, h),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

    rep_df = pd.DataFrame([
        {'representation': 'Per-task emb', 'value': 256, 'kind': 'dim'},
        {'representation': 'Concat emb', 'value': concat_summary['concatenated_embedding_dim'], 'kind': 'dim'},
        {'representation': 'Fusion-ready subjects', 'value': concat_summary['n_subjects'], 'kind': 'subjects'},
    ])
    colors = {'dim': '#f28e2b', 'subjects': '#4e79a7'}
    bar_colors = [colors[k] for k in rep_df['kind']]
    axes[2].bar(rep_df['representation'], rep_df['value'], color=bar_colors)
    axes[2].set_title('Fusion-Ready Output')
    axes[2].set_xlabel('Output summary')
    axes[2].set_ylabel('Value')
    axes[2].set_ylim(0, 850)
    for idx, (_, row) in enumerate(rep_df.iterrows()):
        axes[2].annotate(f"{int(row['value'])}", (idx, row['value']),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

    fig.suptitle('WearGait Update: Task-Aware Training to Fusion-Ready Gait Embedding', fontsize=20, y=1.03)
    fig.text(0.5, -0.02,
             'Separate TCN encoders were trained for SelfPace, HurriedPace, and TUG; '
             'the resulting subject-level embeddings were concatenated into one gait representation for multimodal fusion.',
             ha='center', fontsize=12)

    plt.tight_layout()
    out_path = out_dir / 'weargait_summary_slide.png'
    plt.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close()

    print(out_path)


if __name__ == '__main__':
    main()
