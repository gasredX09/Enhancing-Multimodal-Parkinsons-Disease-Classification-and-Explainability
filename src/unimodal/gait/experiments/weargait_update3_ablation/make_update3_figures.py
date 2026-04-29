from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create presentation-ready Update-3 gait figures from a completed run directory.')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    return parser.parse_args()


def load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / 'summary_metrics.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing summary metrics: {path}')
    return pd.read_csv(path)


def make_representation_best_bar(best_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = best_df.sort_values('mean_auc', ascending=True).copy()
    plot_df['label'] = plot_df['representation'] + ' | ' + plot_df['model']

    sns.set_theme(style='whitegrid', context='talk')
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_df, x='mean_auc', y='label', hue='representation', dodge=False, palette='Set2', ax=ax)
    ax.set_xlim(0.55, 0.85)
    ax.set_xlabel('Mean AUC')
    ax.set_ylabel('')
    ax.set_title('Best Update-3 Model per Gait Representation')
    if ax.legend_:
        ax.legend_.remove()
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(f'{width:.3f}', (width, patch.get_y() + patch.get_height() / 2), ha='left', va='center', fontsize=9, xytext=(5, 0), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def make_ci_forest(best_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = best_df.sort_values('mean_auc', ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    low = plot_df['mean_auc'] - plot_df['ci95_auc_lo']
    high = plot_df['ci95_auc_hi'] - plot_df['mean_auc']

    sns.set_theme(style='whitegrid', context='talk')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(plot_df['mean_auc'], y, xerr=[low, high], fmt='o', color='#1f5aa6', ecolor='#7aa6d8', elinewidth=2, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['representation'] + ' | ' + plot_df['model'])
    ax.set_xlabel('Mean AUC with 95% Interval')
    ax.set_title('Update-3 AUC Stability Across Best Gait Configurations')
    ax.set_xlim(0.55, 0.93)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def make_improvement_plot(df: pd.DataFrame, out_path: Path) -> None:
    raw = df[df['model'] == 'lr_l2'][['representation', 'mean_auc']].rename(columns={'mean_auc': 'raw_auc'})
    best = df.sort_values(['representation', 'mean_auc'], ascending=[True, False]).groupby('representation').head(1)[['representation', 'model', 'mean_auc']].rename(columns={'mean_auc': 'best_auc', 'model': 'best_model'})
    merged = best.merge(raw, on='representation', how='left')
    merged['gain'] = merged['best_auc'] - merged['raw_auc']
    merged = merged.sort_values('gain', ascending=False)

    sns.set_theme(style='whitegrid', context='talk')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=merged, x='representation', y='gain', palette='mako', ax=ax)
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_ylabel('AUC Gain vs Raw lr_l2 Baseline')
    ax.set_xlabel('Representation')
    ax.set_title('How Much Feature Reduction / Regularization Helped')
    for i, row in merged.reset_index(drop=True).iterrows():
        ax.annotate(f"{row['gain']:.3f}\n{row['best_model']}", (i, row['gain']), ha='center', va='bottom', fontsize=8, xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def make_summary_slide(best_df: pd.DataFrame, out_path: Path) -> None:
    overall = best_df.sort_values('mean_auc', ascending=False).iloc[0]
    tug = best_df[best_df['representation'] == 'TUG'].iloc[0]
    all3 = best_df[best_df['representation'] == 'All3'].iloc[0]
    sp_tug = best_df[best_df['representation'] == 'SelfPace+TUG'].iloc[0]

    sns.set_theme(style='white')
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], width_ratios=[1.3, 1, 1])

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title = 'WearGait Update-3: Gait-Only Ablation Summary'
    body = (
        f"Best overall: {overall['representation']} + {overall['model']}\n"
        f"AUC {overall['mean_auc']:.3f} | F1 {overall['mean_f1']:.3f} | Accuracy {overall['mean_accuracy']:.3f}\n\n"
        f"TUG remains the strongest representation.\n"
        f"All-3 improves with feature selection but stays below TUG.\n"
        f"SelfPace+TUG is promising, but also stays below TUG."
    )
    ax_title.text(0.02, 0.82, title, fontsize=24, fontweight='bold', ha='left', va='top')
    ax_title.text(0.02, 0.48, body, fontsize=16, ha='left', va='top')

    cards = [
        ('TUG', tug, '#d6eaf8'),
        ('All3', all3, '#d5f5e3'),
        ('SelfPace+TUG', sp_tug, '#fdebd0'),
    ]
    for idx, (label, row, color) in enumerate(cards):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_facecolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.05, 0.88, label, fontsize=18, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.66, f"Best model: {row['model']}", fontsize=13, transform=ax.transAxes)
        ax.text(0.05, 0.48, f"AUC: {row['mean_auc']:.3f}", fontsize=15, transform=ax.transAxes)
        ax.text(0.05, 0.30, f"F1: {row['mean_f1']:.3f}", fontsize=15, transform=ax.transAxes)
        ax.text(0.05, 0.12, f"Acc: {row['mean_accuracy']:.3f}", fontsize=15, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def write_manifest(best_df: pd.DataFrame, output_dir: Path) -> None:
    manifest = {
        'best_overall': best_df.sort_values('mean_auc', ascending=False).iloc[0].to_dict(),
        'best_per_representation': best_df.to_dict(orient='records'),
        'figures': {
            'best_per_representation_auc_png': str(output_dir / 'best_per_representation_auc.png'),
            'auc_ci_forest_png': str(output_dir / 'auc_ci_forest.png'),
            'improvement_vs_raw_png': str(output_dir / 'improvement_vs_raw.png'),
            'update3_summary_slide_png': str(output_dir / 'update3_summary_slide.png'),
        },
    }
    (output_dir / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.run_dir / 'presentation_figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(args.run_dir)
    best_df = (
        df.sort_values(['representation', 'mean_auc'], ascending=[True, False])
        .groupby('representation')
        .head(1)
        .reset_index(drop=True)
    )

    make_representation_best_bar(best_df, output_dir / 'best_per_representation_auc.png')
    make_ci_forest(best_df, output_dir / 'auc_ci_forest.png')
    make_improvement_plot(df, output_dir / 'improvement_vs_raw.png')
    make_summary_slide(best_df, output_dir / 'update3_summary_slide.png')
    write_manifest(best_df, output_dir)

    print(json.dumps({
        'run_dir': str(args.run_dir),
        'output_dir': str(output_dir),
        'best_overall': best_df.sort_values('mean_auc', ascending=False).iloc[0].to_dict(),
    }, indent=2))


if __name__ == '__main__':
    main()
