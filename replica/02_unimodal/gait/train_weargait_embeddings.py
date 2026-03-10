from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
TASKS_DEFAULT = ["SelfPace", "HurriedPace", "TUG"]


# Core channel subset for v1 to reduce heterogeneity but keep gait signal richness.
CHANNELS_V1 = [
    "LowerBack_Acc_X", "LowerBack_Acc_Y", "LowerBack_Acc_Z",
    "LowerBack_Gyr_X", "LowerBack_Gyr_Y", "LowerBack_Gyr_Z",
    "L_Ankle_Acc_X", "L_Ankle_Acc_Y", "L_Ankle_Acc_Z",
    "L_Ankle_Gyr_X", "L_Ankle_Gyr_Y", "L_Ankle_Gyr_Z",
    "R_Ankle_Acc_X", "R_Ankle_Acc_Y", "R_Ankle_Acc_Z",
    "R_Ankle_Gyr_X", "R_Ankle_Gyr_Y", "R_Ankle_Gyr_Z",
    "LTotalForce", "RTotalForce", "LCoP_X", "LCoP_Y", "RCoP_X", "RCoP_Y",
]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]  # project/replica
    index_csv = repo_root / "outputs" / "unimodal_gait" / "weargait_index.csv"
    output_dir = repo_root / "outputs" / "unimodal_gait" / "weargait_dl_embeddings"
    return index_csv, output_dir


def parse_time_column(s: pd.Series) -> np.ndarray:
    # Values are usually like "0 sec", "0.01 sec".
    cleaned = s.astype(str).str.replace("sec", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").to_numpy(dtype=np.float32)


def preprocess_frame(df: pd.DataFrame, channels: Sequence[str]) -> np.ndarray:
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing channels: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    x = df.loc[:, channels].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    # Bidirectional fill then zero for trailing full-NaN columns.
    x = x.ffill().bfill().fillna(0.0)
    return x.to_numpy(dtype=np.float32)


def maybe_resample(x: np.ndarray, t: np.ndarray, target_hz: float) -> np.ndarray:
    if x.shape[0] < 4:
        return x

    valid = np.isfinite(t)
    if valid.sum() < 4:
        return x

    t = t[valid]
    x = x[valid]
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return x

    median_dt = float(np.median(dt))
    current_hz = 1.0 / median_dt
    if abs(current_hz - target_hz) < 1.0:
        return x

    t_new = np.arange(float(t[0]), float(t[-1]) + 1e-8, 1.0 / target_hz, dtype=np.float32)
    out = np.zeros((len(t_new), x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_new, t, x[:, c]).astype(np.float32)
    return out


def make_windows(x: np.ndarray, window_samples: int, step_samples: int) -> np.ndarray:
    if x.shape[0] < window_samples:
        pad = np.zeros((window_samples - x.shape[0], x.shape[1]), dtype=np.float32)
        x = np.vstack([x, pad])

    windows = []
    for start in range(0, x.shape[0] - window_samples + 1, step_samples):
        windows.append(x[start:start + window_samples])

    if not windows:
        windows.append(x[:window_samples])

    return np.stack(windows, axis=0)  # (N, T, C)


@dataclass
class WindowRecord:
    subject_id: str
    task: str
    label: int


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(np.transpose(x, (0, 2, 1))).float()  # (N, C, T)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float):
        super().__init__()
        k = 3
        p = dilation * (k - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=p, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=p, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.skip(x))


class GaitTCN(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        chs = [64, 64, 128, 128]
        dils = [1, 2, 4, 8]
        layers: List[nn.Module] = []
        cur = in_ch
        for out_ch, dil in zip(chs, dils):
            layers.append(ConvBlock(cur, out_ch, dilation=dil, dropout=dropout))
            cur = out_ch
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(cur, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )
        self.cls = nn.Linear(emb_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.pool(h).squeeze(-1)
        emb = self.proj(h)
        logits = self.cls(emb)
        return logits, emb


def train_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def eval_epoch(model, loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            losses.append(float(loss.item()))
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    return metrics


def infer_embeddings(model, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    ds = WindowDataset(x=x, y=np.zeros((len(x),), dtype=np.int64))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embs: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            _, eb = model(xb)
            embs.append(eb.cpu().numpy())
    if not embs:
        return np.zeros((0, model.cls.in_features), dtype=np.float32)
    return np.vstack(embs)


def parse_args() -> argparse.Namespace:
    index_csv, output_dir = build_default_paths()
    parser = argparse.ArgumentParser(description="Train WearGait TCN and export subject-level embeddings")
    parser.add_argument("--index-csv", type=Path, default=index_csv)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--target-hz", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.index_csv.exists():
        raise FileNotFoundError(
            f"Index CSV not found: {args.index_csv}. Run prepare_weargait_index.py first."
        )

    df = pd.read_csv(args.index_csv)
    df = df[df["task"].isin(args.tasks)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after task filtering.")

    window_samples = int(args.window_sec * args.target_hz)
    step_samples = max(1, int(window_samples * (1.0 - args.overlap)))

    windows_list: List[np.ndarray] = []
    records: List[WindowRecord] = []

    for row in df.itertuples(index=False):
        csv_path = Path(row.file_path)
        # WearGait CSVs can have mixed inferred dtypes in non-modeled columns.
        # Disable chunked inference to avoid repetitive DtypeWarning noise.
        raw = pd.read_csv(csv_path, low_memory=False)
        if "Time" not in raw.columns:
            continue

        t = parse_time_column(raw["Time"])
        try:
            x = preprocess_frame(raw, CHANNELS_V1)
        except ValueError:
            continue

        x = maybe_resample(x, t=t, target_hz=args.target_hz)
        xw = make_windows(x, window_samples=window_samples, step_samples=step_samples)

        for i in range(xw.shape[0]):
            windows_list.append(xw[i])
            records.append(WindowRecord(subject_id=str(row.subject_id), task=str(row.task), label=int(row.label)))

    if not windows_list:
        raise ValueError("No valid windows generated. Check channels/tasks and source files.")

    X = np.stack(windows_list, axis=0).astype(np.float32)  # (N, T, C)
    y = np.array([r.label for r in records], dtype=np.int64)
    groups = np.array([r.subject_id for r in records])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_rows: List[Dict] = []
    subject_emb_store: Dict[str, List[np.ndarray]] = {}

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        scaler = StandardScaler()
        scaler.fit(X_tr.reshape(-1, X_tr.shape[-1]))

        def transform(arr: np.ndarray) -> np.ndarray:
            flat = arr.reshape(-1, arr.shape[-1])
            flat = scaler.transform(flat)
            return flat.reshape(arr.shape).astype(np.float32)

        X_tr = transform(X_tr)
        X_va = transform(X_va)

        ds_tr = WindowDataset(X_tr, y_tr)
        ds_va = WindowDataset(X_va, y_va)

        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

        model = GaitTCN(in_ch=X.shape[-1], emb_dim=args.emb_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_f1 = -1.0
        best_wait = 0

        for _ in range(args.epochs):
            train_epoch(model, dl_tr, optimizer, criterion, device)
            m = eval_epoch(model, dl_va, criterion, device)
            f1 = m["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_wait = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                best_wait += 1
                if best_wait >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = eval_epoch(model, dl_va, criterion, device)
        fold_rows.append({"fold": fold, **metrics, "n_val": int(len(va))})

        # Export validation-window embeddings (OOF style), then average by subject.
        emb_va = infer_embeddings(model, X_va, batch_size=args.batch_size, device=device)
        for idx_local, emb in enumerate(emb_va):
            subject_id = groups[va][idx_local]
            subject_emb_store.setdefault(subject_id, []).append(emb.astype(np.float32))

        torch.save(model.state_dict(), args.output_dir / f"fold_{fold}_best_model.pt")

    # Aggregate subject embeddings and labels
    subjects = sorted(subject_emb_store.keys())
    emb_mat = np.stack([np.mean(np.stack(subject_emb_store[s], axis=0), axis=0) for s in subjects], axis=0)

    subj_labels = []
    for sid in subjects:
        values = df.loc[df["subject_id"] == sid, "label"].unique().tolist()
        subj_labels.append(int(values[0]) if values else 0)

    out_npz = args.output_dir / "weargait_subject_embeddings.npz"
    np.savez_compressed(
        out_npz,
        subject_ids=np.array(subjects),
        y=np.array(subj_labels, dtype=np.int64),
        X_emb=emb_mat.astype(np.float32),
    )

    pd.DataFrame(fold_rows).to_csv(args.output_dir / "cv_metrics.csv", index=False)

    summary = {
        "n_windows": int(X.shape[0]),
        "n_subjects": int(len(np.unique(groups))),
        "n_channels": int(X.shape[-1]),
        "window_sec": args.window_sec,
        "target_hz": args.target_hz,
        "tasks": args.tasks,
        "embedding_dim": args.emb_dim,
        "mean_acc": float(np.nanmean([r["acc"] for r in fold_rows])),
        "mean_f1": float(np.nanmean([r["f1"] for r in fold_rows])),
        "mean_auc": float(np.nanmean([r["auc"] for r in fold_rows])),
        "outputs": {
            "metrics_csv": str(args.output_dir / "cv_metrics.csv"),
            "embeddings_npz": str(out_npz),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {args.output_dir / 'cv_metrics.csv'}")
    print(f"  {out_npz}")
    print(f"  {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
