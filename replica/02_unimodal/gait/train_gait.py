"""
Unimodal Gait Classification for Parkinson's Disease Severity
Uses TCN architecture with supervised learning on IMU sensor data.

Binary classification task:
- Class 0: Mild PD (H&Y score ≤ 2.0)
- Class 1: Moderate/Severe PD (H&Y score > 2.0)

Dataset: Figshare PDFE (freezing of gait patients only)
"""

import os
import re
import random
from pathlib import Path
from io import StringIO
import pickle
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== CONFIG ====================
# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # project/replica/
DATA_DIR = BASE_DIR / "data" / "gait" / "figshare"
IMU_DIR = DATA_DIR / "IMU"
LABELS_FILE = DATA_DIR / "PDFEinfo.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "unimodal_gait"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data parameters
SAMPLE_RATE = 100  # Hz
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)  # 200
WINDOW_STEP = WINDOW_SAMPLES // 2  # 50% overlap

# Training parameters
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5
N_FOLDS = 5

# Model parameters
TCN_CHANNELS = [64, 64, 128, 128]
DILATIONS = [1, 2, 4, 8]
KERNEL_SIZE = 3
DROPOUT = 0.3

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")


# ==================== UTILS ====================
def seed_everything(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_subject_id(filename):
    """Extract subject ID from filename (e.g., SUB01, GaCo01, GaPt01, JuCo01)."""
    base = Path(filename).stem
    match = re.match(r'^(SUB\d+|GaCo\d+|GaPt\d+|JuCo\d+)', base, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return base.split('_')[0].upper()


def is_float(s):
    """Check if string can be converted to float."""
    try:
        float(s)
        return True
    except:
        return False


def read_gait_file(path):
    """
    Read gait data from .txt or .csv file.
    Returns array with shape (timesteps, 6) containing only IMU sensor channels:
    - Columns 0-2: ACC ML/AP/SI (accelerometer)
    - Columns 3-5: GYR ML/AP/SI (gyroscope)
    
    Drops: Frame #, Time, and Freezing flag columns
    """
    try:
        arr = np.loadtxt(path)
    except Exception:
        # Fallback: parse line-by-line, keeping only numeric entries
        with open(path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        numeric_lines = []
        for line in lines:
            parts = line.strip().split()
            numeric_parts = [tok for tok in parts if is_float(tok)]
            if len(numeric_parts) >= 2:
                numeric_lines.append(" ".join(numeric_parts))
        
        if not numeric_lines:
            raise ValueError(f"No numeric data in {path}")
        
        arr = np.loadtxt(StringIO("\n".join(numeric_lines)))
    
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    # Expected format: Frame#, Time, ACC_ML, ACC_AP, ACC_SI, GYR_ML, GYR_AP, GYR_SI, Freezing_flag
    # Keep only columns 2-7 (6 IMU sensor channels)
    if arr.shape[1] >= 8:
        return arr[:, 2:8].astype(np.float32)
    else:
        # Fallback for different format: drop first column only
        return arr[:, 1:].astype(np.float32)


def sliding_windows(data, win_len=WINDOW_SAMPLES, step=WINDOW_STEP):
    """
    Create sliding windows from time-series data.
    Returns list of (start, end, window_data) tuples.
    """
    L = data.shape[0]
    
    # Pad if too short
    if L < win_len:
        pad = np.zeros((win_len - L, data.shape[1]), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=0)
        L = win_len
    
    windows = []
    for start in range(0, L - win_len + 1, step):
        end = start + win_len
        windows.append((start, end, data[start:end]))
    
    return windows


# ==================== DATASET ====================
class GaitDataset(Dataset):
    """PyTorch dataset for gait windows."""
    
    def __init__(self, windows, labels):
        """
        Args:
            windows: List of numpy arrays, each (channels, time)
            labels: List of labels (0 or 1)
        """
        self.windows = windows
        self.labels = labels
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ==================== MODEL ====================
class ConvBlock(nn.Module):
    """Temporal convolutional block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = out + residual
        out = self.relu_out(out)
        
        return out


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network for gait classification."""
    
    def __init__(self, in_channels, tcn_channels, dilations, kernel_size=3, dropout=0.3, n_classes=2):
        super().__init__()
        
        # TCN encoder
        layers = []
        current_channels = in_channels
        for out_ch, dilation in zip(tcn_channels, dilations):
            layers.append(ConvBlock(current_channels, out_ch, kernel_size, dilation, dropout))
            current_channels = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        features = self.encoder(x)
        pooled = self.pool(features).squeeze(-1)  # (batch, channels)
        logits = self.classifier(pooled)
        return logits
    
    def extract_features(self, x):
        """Extract features without classification."""
        features = self.encoder(x)
        pooled = self.pool(features).squeeze(-1)
        return pooled


# ==================== TRAINING ====================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, all_preds, all_probs, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device, fold_dir):
    """Train model with early stopping."""
    best_val_acc = 0
    best_epoch = 0
    no_improve = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, fold_dir / 'best_model.pth')
            print(f"✓ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nBest Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    
    return history, best_val_acc


# ==================== MAIN ====================
def main():
    seed_everything()
    
    print("="*60)
    print("GAIT-BASED PARKINSON'S DISEASE CLASSIFICATION")
    print("="*60)
    
    # ========== Load Labels ==========
    print("\n1. Loading labels...")
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")
    
    # Try different encodings for Windows-generated CSV files
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            labels_df = pd.read_csv(LABELS_FILE, encoding=encoding, sep=';')
            print(f"✓ Loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Could not read {LABELS_FILE} with any standard encoding")
    
    print(f"Loaded {len(labels_df)} subject labels")
    print(f"Columns: {labels_df.columns.tolist()[:10]}")
    
    # This dataset contains only PD patients (PDFE01-PDFE35) with no HC
    # Use H&Y (Hoehn & Yahr) score for binary classification:
    # - Mild PD (H&Y = 2.0): label 0
    # - Moderate/Severe PD (H&Y >= 3.0): label 1
    
    if 'ID' not in labels_df.columns:
        raise ValueError(f"Expected 'ID' column not found. Columns: {labels_df.columns.tolist()}")
    
    hy_col = 'Session 1 - H&Y (score)'
    if hy_col not in labels_df.columns:
        raise ValueError(f"Expected '{hy_col}' column not found")
    
    subject_labels = {}
    for _, row in labels_df.iterrows():
        pdfe_id = str(row['ID']).strip().upper()
        hy_score = row[hy_col]
        
        if pd.isna(hy_score):
            continue
        
        # Map PDFE## to SUB## (e.g., PDFE01 -> SUB01)
        if pdfe_id.startswith('PDFE'):
            num = pdfe_id.replace('PDFE', '')
            sub_id = f'SUB{num}'
            
            # Binary classification by H&Y severity
            label = 0 if hy_score <= 2.0 else 1
            subject_labels[sub_id] = label
            subject_labels[pdfe_id] = label  # Store both mappings
    
    print(f"Label distribution: Mild PD (H&Y≤2)={sum(v==0 for v in subject_labels.values())//2}, "
          f"Moderate/Severe PD (H&Y>2)={sum(v==1 for v in subject_labels.values())//2}")
    
    # ========== Load Gait Data ==========
    print("\n2. Loading gait data...")
    # Only load .txt files (.csv files in this dataset are actually Excel format)
    gait_files = sorted(list(IMU_DIR.glob('*.txt')))
    print(f"Found {len(gait_files)} gait files")
    
    if len(gait_files) == 0:
        raise RuntimeError(f"No gait files found in {IMU_DIR}")
    
    # Process files and create windows
    all_windows = []
    all_labels = []
    all_subjects = []
    failed_files = []
    subjects_with_windows = set()
    
    for file_path in tqdm(gait_files, desc="Processing files"):
        try:
            # Read data
            data = read_gait_file(file_path)
            subject_id = parse_subject_id(file_path.name)
            
            # Get label
            if subject_id not in subject_labels:
                # Try partial matching
                found = False
                for subj_key in subject_labels.keys():
                    if subject_id in subj_key or subj_key in subject_id:
                        subject_id = subj_key
                        found = True
                        break
                
                if not found:
                    continue  # Skip if no label
            
            label = subject_labels[subject_id]
            
            # Create windows
            windows = sliding_windows(data)
            
            for start, end, window in windows:
                # Transpose to (channels, time)
                window_t = window.T.copy()
                
                # Validate shape
                if window_t.shape != (6, WINDOW_SAMPLES):
                    print(f"WARNING: Window from {file_path.name} has shape {window_t.shape}, expected (6, {WINDOW_SAMPLES})")
                    continue  # Skip malformed windows
                
                all_windows.append(window_t)
                all_labels.append(label)
                all_subjects.append(subject_id)
                subjects_with_windows.add(subject_id)
        
        except Exception as e:
            failed_files.append((file_path.name, str(e)))
    
    print(f"\n✓ Loaded {len(all_windows)} windows from {len(subjects_with_windows)} subjects")
    print(f"✗ Failed to load {len(failed_files)} files")
    
    if len(failed_files) > 0:
        print("\nFailed files (first 5):")
        for fname, error in failed_files[:5]:
            print(f"  - {fname}: {error}")
    
    # Convert to arrays
    if len(all_windows) == 0:
        raise RuntimeError(
            f"No data loaded! Check that:\n"
            f"1. .txt files exist in {IMU_DIR}\n"
            f"2. Subject IDs in filenames match PDFEinfo.csv (SUB01-SUB35)\n"
            f"3. Files have valid numeric data\n"
            f"Failed files: {len(failed_files)}/{len(gait_files)}"
        )

    # Diagnostic: check for shape consistency before conversion
    shapes_found = set(w.shape for w in all_windows[:100])
    if len(shapes_found) > 1:
        print(f"WARNING: Found {len(shapes_found)} different window shapes in sample: {shapes_found}")

        # Count occurrences across all windows
        shape_counts = {}
        for w in all_windows:
            shape_counts[w.shape] = shape_counts.get(w.shape, 0) + 1

        print("Shape distribution:")
        for shape, count in sorted(shape_counts.items()):
            print(f"  {shape}: {count} windows")
    
    # Use stack for better error messages if shapes don't match
    try:
        all_windows = np.stack(all_windows, axis=0).astype(np.float32)
    except ValueError as e:
        print(f"ERROR stacking windows: {e}")
        # Find the first mismatch
        expected_shape = all_windows[0].shape
        for i, w in enumerate(all_windows):
            if w.shape != expected_shape:
                print(f"Window {i} has shape {w.shape}, expected {expected_shape}")
                if i < 10:  # Show first 10 mismatches
                    continue
                else:
                    break
        raise
    
    all_labels = np.array(all_labels, dtype=np.int64)
    all_subjects = np.array(all_subjects)
    
    print(f"\nData shape: {all_windows.shape}")
    print(f"Labels: Mild PD={sum(all_labels==0)}, Moderate/Severe PD={sum(all_labels==1)}")
    
    # Get number of channels
    n_channels = all_windows.shape[1]
    print(f"Number of channels: {n_channels}")
    
    # ========== Normalize Data ==========
    print("\n3. Normalizing data...")
    
    # Flatten for normalization
    n_samples, n_ch, n_time = all_windows.shape
    X_flat = all_windows.reshape(n_samples, -1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_scaled.reshape(n_samples, n_ch, n_time)
    
    # Save scaler
    with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {OUTPUT_DIR / 'scaler.pkl'}")
    
    # ========== Cross-Validation ==========
    print(f"\n4. Starting {N_FOLDS}-fold cross-validation...")
    
    # Get unique subjects for stratified group k-fold
    unique_subjects = np.unique(all_subjects)
    subject_to_label = {subj: all_labels[all_subjects == subj][0] for subj in unique_subjects}
    
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    fold_results = []
    all_fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_scaled, all_labels, groups=all_subjects)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")
        
        fold_dir = OUTPUT_DIR / f'fold_{fold+1}'
        fold_dir.mkdir(exist_ok=True)
        
        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]
        
        print(f"Train: {len(X_train)} samples (Mild={sum(y_train==0)}, Moderate/Severe={sum(y_train==1)})")
        print(f"Val: {len(X_val)} samples (Mild={sum(y_val==0)}, Moderate/Severe={sum(y_val==1)})")
        
        # Create datasets
        train_dataset = GaitDataset(X_train, y_train)
        val_dataset = GaitDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Create model
        model = TCNClassifier(
            in_channels=n_channels,
            tcn_channels=TCN_CHANNELS,
            dilations=DILATIONS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
            n_classes=2
        ).to(DEVICE)
        
        # Calculate class weights to handle imbalance
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        
        print(f"Class weights: Mild PD={class_weights[0]:.4f}, Moderate/Severe PD={class_weights[1]:.4f}")
        
        # Loss and optimizer with class weighting
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Train
        history, best_val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            EPOCHS, PATIENCE, DEVICE, fold_dir
        )
        
        all_fold_histories.append(history)
        
        # Load best model for evaluation
        checkpoint = torch.load(fold_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        print("\nFinal evaluation on validation set...")
        val_loss, val_acc, val_preds, val_probs, val_labels = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Compute metrics
        precision = precision_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        auc = roc_auc_score(val_labels, val_probs)
        
        print(f"\nMetrics:")
        print(f"  Accuracy:  {val_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Mild PD', 'Moderate/Severe PD'], 
                    yticklabels=['Mild PD', 'Moderate/Severe PD'])
        plt.title(f'Confusion Matrix - Fold {fold+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(fold_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results
        fold_results.append({
            'fold': fold + 1,
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'best_epoch': checkpoint['epoch'] + 1
        })
        
        # Save classification report
        report = classification_report(val_labels, val_preds, 
                                      target_names=['Mild PD', 'Moderate/Severe PD'])
        with open(fold_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\nClassification Report:\n{report}")
    
    # ========== Aggregate Results ==========
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(fold_results)
    print("\nPer-fold results:")
    print(results_df.to_string(index=False))
    
    print("\nMean ± Std:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        print(f"  {metric.capitalize():10s}: {mean:.4f} ± {std:.4f}")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / 'cv_results.csv', index=False)
    
    # Save summary
    summary = {
        'n_folds': N_FOLDS,
        'n_samples': len(all_windows),
        'n_subjects': len(subjects_with_windows),
        'n_channels': n_channels,
        'metrics': {
            metric: {
                'mean': float(results_df[metric].mean()),
                'std': float(results_df[metric].std())
            }
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {OUTPUT_DIR}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
