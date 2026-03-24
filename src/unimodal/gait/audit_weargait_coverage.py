from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from train_weargait_embeddings import CHANNELS_V1, parse_time_column, preprocess_frame, maybe_resample, make_windows


def build_default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    index_csv = repo_root / 'outputs' / 'unimodal_gait' / 'weargait_index.csv'
    output_dir = repo_root / 'outputs' / 'unimodal_gait' / 'weargait_audit'
    return index_csv, output_dir


def parse_args() -> argparse.Namespace:
    index_csv, output_dir = build_default_paths()
    parser = argparse.ArgumentParser(description='Audit WearGait file coverage and drop reasons.')
    parser.add_argument('--index-csv', type=Path, default=index_csv)
    parser.add_argument('--output-dir', type=Path, default=output_dir)
    parser.add_argument('--window-sec', type=float, default=6.0)
    parser.add_argument('--target-hz', type=float, default=100.0)
    parser.add_argument('--overlap', type=float, default=0.5)
    return parser.parse_args()


def audit_file(csv_path: Path, channels: Sequence[str], window_samples: int, step_samples: int, target_hz: float) -> dict:
    result = {
        'file_path': str(csv_path),
        'status': 'ok',
        'reason': '',
        'n_columns': 0,
        'n_rows': 0,
        'n_missing_channels': 0,
        'missing_channels': '',
        'n_valid_time': 0,
        'resampled_rows': 0,
        'n_windows': 0,
    }
    try:
        raw = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        result['status'] = 'drop'
        result['reason'] = f'read_error: {type(exc).__name__}'
        return result

    result['n_columns'] = len(raw.columns)
    result['n_rows'] = len(raw)

    if 'Time' not in raw.columns:
        result['status'] = 'drop'
        result['reason'] = 'missing_time_column'
        return result

    missing = [c for c in channels if c not in raw.columns]
    result['n_missing_channels'] = len(missing)
    result['missing_channels'] = ';'.join(missing)
    if missing:
        result['status'] = 'drop'
        result['reason'] = 'missing_required_channels'
        return result

    t = parse_time_column(raw['Time'])
    result['n_valid_time'] = int(np.isfinite(t).sum())
    if result['n_valid_time'] < 4:
        result['status'] = 'drop'
        result['reason'] = 'insufficient_valid_time'
        return result

    try:
        x = preprocess_frame(raw, channels)
    except Exception as exc:
        result['status'] = 'drop'
        result['reason'] = f'preprocess_error: {type(exc).__name__}'
        return result

    try:
        xr = maybe_resample(x, t=t, target_hz=target_hz)
        result['resampled_rows'] = int(xr.shape[0])
        xw = make_windows(xr, window_samples=window_samples, step_samples=step_samples)
        result['n_windows'] = int(xw.shape[0])
    except Exception as exc:
        result['status'] = 'drop'
        result['reason'] = f'window_error: {type(exc).__name__}'
        return result

    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.index_csv.exists():
        raise FileNotFoundError(f'Index CSV not found: {args.index_csv}')

    df = pd.read_csv(args.index_csv)
    window_samples = int(args.window_sec * args.target_hz)
    step_samples = max(1, int(window_samples * (1.0 - args.overlap)))

    rows = []
    for row in df.itertuples(index=False):
        rec = audit_file(Path(row.file_path), CHANNELS_V1, window_samples, step_samples, args.target_hz)
        rec['subject_id'] = str(row.subject_id)
        rec['task'] = str(row.task)
        rec['label'] = int(row.label)
        rows.append(rec)

    audit_df = pd.DataFrame(rows)
    audit_csv = args.output_dir / 'weargait_file_audit.csv'
    audit_df.to_csv(audit_csv, index=False)

    usable_df = audit_df[audit_df['status'] == 'ok'].copy()
    subject_task_counts = usable_df.groupby('task')['subject_id'].nunique().to_dict()
    reason_counts = audit_df[audit_df['status'] != 'ok']['reason'].value_counts().to_dict()

    missing_channel_counts = {}
    dropped_missing = audit_df[audit_df['reason'] == 'missing_required_channels']
    if not dropped_missing.empty:
        for item in dropped_missing['missing_channels']:
            for ch in str(item).split(';'):
                if ch:
                    missing_channel_counts[ch] = missing_channel_counts.get(ch, 0) + 1

    summary = {
        'index_rows': int(len(audit_df)),
        'index_subjects': int(audit_df['subject_id'].nunique()),
        'usable_rows': int(len(usable_df)),
        'usable_subjects': int(usable_df['subject_id'].nunique()),
        'usable_subjects_by_task': subject_task_counts,
        'drop_reason_counts': reason_counts,
        'missing_channel_counts': dict(sorted(missing_channel_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        'outputs': {
            'audit_csv': str(audit_csv),
        },
    }

    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
