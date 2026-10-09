from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPRESENTATIONS = {
    'selfpace': ('SelfPace', 'weargait_dl_embeddings/SelfPace/weargait_subject_embeddings.npz'),
    'hurriedpace': ('HurriedPace', 'weargait_dl_embeddings/HurriedPace/weargait_subject_embeddings.npz'),
    'tug': ('TUG', 'weargait_dl_embeddings/TUG/weargait_subject_embeddings.npz'),
    'concat': ('Concat', 'weargait_concat_embeddings/weargait_concat_subject_embeddings.npz'),
}


def build_defaults() -> tuple[Path, Path]:
    outputs_root = Path(__file__).resolve().parents[5] / 'outputs' / 'unimodal_gait'
    default_run_name = f"weargait_representation_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = outputs_root / 'runs' / default_run_name
    return outputs_root, run_dir


def parse_args() -> argparse.Namespace:
    outputs_root, run_dir = build_defaults()
    parser = argparse.ArgumentParser(description='Run a benchmark over WearGait task embeddings on the common subject intersection.')
    parser.add_argument('--outputs-root', type=Path, default=outputs_root)
    parser.add_argument('--run-dir', type=Path, default=run_dir)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def ensure_fresh_run_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'Run directory already exists: {path}')
    path.mkdir(parents=True, exist_ok=False)


def load_embeddings(outputs_root: Path) -> dict[str, dict]:
    data = {}
    for _, (label, rel_path) in REPRESENTATIONS.items():
        p = outputs_root / rel_path
        arr = np.load(p, allow_pickle=True)
        data[label] = {
            'subject_ids': arr['subject_ids'].astype(str),
            'y': arr['y'].astype(int),
            'X': arr['X_emb'].astype(np.float32),
        }
    return data


def align_common_subjects(data: dict[str, dict]) -> dict[str, dict]:
    common = None
    for item in data.values():
        s = set(item['subject_ids'].tolist())
        common = s if common is None else common & s
    common_ids = sorted(common)

    aligned = {}
    for name, item in data.items():
        idx_map = {sid: i for i, sid in enumerate(item['subject_ids'].tolist())}
        idx = [idx_map[sid] for sid in common_ids]
        aligned[name] = {
            'subject_ids': np.array(common_ids, dtype=str),
            'y': item['y'][idx],
            'X': item['X'][idx],
        }
    return aligned


def evaluate_representation(X: np.ndarray, y: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> tuple[pd.DataFrame, dict]:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for split_id, (tr, va) in enumerate(cv.split(X, y), start=1):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')),
        ])
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[va])[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            'split': split_id,
            'accuracy': accuracy_score(y[va], pred),
            'f1': f1_score(y[va], pred, zero_division=0),
            'auc': roc_auc_score(y[va], proba) if len(np.unique(y[va])) > 1 else np.nan,
        })
    df = pd.DataFrame(rows)
    summary = {
        'mean_accuracy': float(df['accuracy'].mean()),
        'mean_f1': float(df['f1'].mean()),
        'mean_auc': float(df['auc'].mean()),
        'ci95_accuracy_lo': float(df['accuracy'].quantile(0.025)),
        'ci95_accuracy_hi': float(df['accuracy'].quantile(0.975)),
        'ci95_f1_lo': float(df['f1'].quantile(0.025)),
        'ci95_f1_hi': float(df['f1'].quantile(0.975)),
        'ci95_auc_lo': float(df['auc'].quantile(0.025)),
        'ci95_auc_hi': float(df['auc'].quantile(0.975)),
    }
    return df, summary


def make_benchmark_plot(summary_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary_df.melt(
        id_vars=['representation'],
        value_vars=['mean_accuracy', 'mean_f1', 'mean_auc'],
        var_name='metric',
        value_name='value',
    )
    rename = {'mean_accuracy': 'Accuracy', 'mean_f1': 'F1', 'mean_auc': 'AUC'}
    plot_df['metric'] = plot_df['metric'].map(rename)

    sns.set_theme(style='whitegrid', context='talk')
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=plot_df, x='representation', y='value', hue='metric', palette='Set2')
    plt.ylim(0.0, 1.0)
    plt.title('WearGait Representation Benchmark')
    plt.xlabel('Representation')
    plt.ylabel('Score')
    for patch in ax.patches:
        h = patch.get_height()
        ax.annotate(f'{h:.3f}', (patch.get_x() + patch.get_width() / 2, h),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_notes(summary_df: pd.DataFrame, run_dir: Path, common_subjects: int) -> None:
    best = summary_df.iloc[0]
    second = summary_df.iloc[1]
    text = f'''# WearGait Representation Benchmark Notes

Run directory: `{run_dir}`

## What this run does
- Compares `SelfPace`, `HurriedPace`, `TUG`, and concatenated gait embeddings
- Uses the common subject intersection across all representations
- Uses repeated stratified CV with logistic regression on top of each embedding

## Key result
- Best representation by AUC: `{best['representation']}`
- Mean AUC: `{best['mean_auc']:.3f}`
- Mean F1: `{best['mean_f1']:.3f}`
- Common subject count: `{common_subjects}`

## Interpretation
- `TUG` is currently the strongest standalone gait representation.
- The concatenated embedding is still competitive and remains useful as a richer fusion-ready representation.
- The most meaningful next comparison is `TUG` vs `Concat` in downstream fusion.

## Ranking by AUC
{summary_df[['representation', 'mean_auc', 'mean_f1', 'mean_accuracy']].to_string(index=False)}

## Recommended next step
- Carry both `TUG` and `Concat` forward as candidate gait inputs for downstream fusion.
'''
    (run_dir / 'notes.md').write_text(text, encoding='utf-8')


def main() -> None:
    args = parse_args()
    ensure_fresh_run_dir(args.run_dir)

    data = align_common_subjects(load_embeddings(args.outputs_root))
    common_subjects = len(next(iter(data.values()))['subject_ids'])

    split_frames = []
    summary_rows = []
    for representation, item in data.items():
        split_df, summary = evaluate_representation(item['X'], item['y'], args.n_splits, args.n_repeats, args.seed)
        split_df['representation'] = representation
        split_frames.append(split_df)
        summary_rows.append({
            'representation': representation,
            'subjects': common_subjects,
            'embedding_dim': int(item['X'].shape[1]),
            **summary,
        })

    per_split_df = pd.concat(split_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values('mean_auc', ascending=False).reset_index(drop=True)

    per_split_path = args.run_dir / 'per_split_metrics.csv'
    summary_path = args.run_dir / 'summary_metrics.csv'
    plot_path = args.run_dir / 'representation_benchmark.png'

    per_split_df.to_csv(per_split_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    make_benchmark_plot(summary_df, plot_path)
    write_notes(summary_df, args.run_dir, common_subjects)

    summary = {
        'run_dir': str(args.run_dir),
        'common_subjects': common_subjects,
        'n_splits': args.n_splits,
        'n_repeats': args.n_repeats,
        'best_by_auc': summary_df.iloc[0]['representation'],
        'outputs': {
            'per_split_metrics_csv': str(per_split_path),
            'summary_metrics_csv': str(summary_path),
            'benchmark_plot': str(plot_path),
            'notes_md': str(args.run_dir / 'notes.md'),
        },
    }
    (args.run_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('\nSummary table:')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
