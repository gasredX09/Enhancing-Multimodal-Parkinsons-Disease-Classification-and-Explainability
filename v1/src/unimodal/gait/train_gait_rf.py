"""
Random Forest Baseline for Gait-Based PD Classification
========================================================

Trains a Random Forest classifier on engineered gait features
as a baseline for comparison with deep learning models.

Task: Binary classification of PDFE severity
- Class 0: Mild PD (H&Y score ≤ 2.0)
- Class 1: Moderate/Severe PD (H&Y score > 2.0)

Features engineered from gait IMU data:
- Stride length variability
- Cadence regularity
- Bilateral asymmetry
- Frequency domain features
- Acceleration/gyroscope statistics

Usage:
    python train_gait_rf.py
    python train_gait_rf.py --n-folds 5 --seed 42
"""

import re
import random
from pathlib import Path
from io import StringIO
import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== CONFIG ====================
BASE_DIR = Path(__file__).resolve().parents[3]  # project/
DATA_DIR = BASE_DIR / "data" / "gait" / "figshare"
IMU_DIR = DATA_DIR / "IMU"
LABELS_FILE = DATA_DIR / "PDFEinfo.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "unimodal_gait" / "Random_Forest_Baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
SEED = 42
N_FOLDS = 5
N_ESTIMATORS = 100
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 5

# Gait signal processing
SAMPLE_RATE = 100  # Hz
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)

print(f"Output directory: {OUTPUT_DIR}")


# ==================== UTILS ====================
def seed_everything(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def parse_subject_id(filename):
    """Extract subject ID from filename."""
    base = Path(filename).stem
    match = re.match(r'^SUB\d+', base, re.IGNORECASE)
    if match:
        return match.group().upper()
    return base.split('_')[0].upper()


def is_float(s):
    """Check if string can be converted to float."""
    try:
        float(s)
        return True
    except:
        return False


def read_gait_file(path):
    """Read gait IMU data from .txt or .csv file."""
    try:
        arr = np.loadtxt(path)
    except Exception:
        # Fallback: parse line-by-line
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
    
    # Extract 6 IMU channels (ACC + GYR)
    if arr.shape[1] >= 8:
        return arr[:, 2:8].astype(np.float32)
    else:
        return arr[:, 1:].astype(np.float32)


# ==================== FEATURE ENGINEERING ====================
def extract_gait_features(data):
    """
    Extract engineered gait features from IMU time-series.
    
    Args:
        data: (timesteps, 6) array [ACC_ML, ACC_AP, ACC_SI, GYR_ML, GYR_AP, GYR_SI]
    
    Returns:
        (features_dict, features_array)
    """
    features = {}
    
    # Acceleration channels (0-2)
    acc = data[:, :3]
    
    # Gyroscope channels (3-5)
    gyr = data[:, 3:]
    
    # 1. Acceleration statistics
    features['acc_mean'] = np.mean(acc, axis=0)
    features['acc_std'] = np.std(acc, axis=0)
    features['acc_min'] = np.min(acc, axis=0)
    features['acc_max'] = np.max(acc, axis=0)
    
    # 2. Gyroscope statistics
    features['gyr_mean'] = np.mean(gyr, axis=0)
    features['gyr_std'] = np.std(gyr, axis=0)
    features['gyr_min'] = np.min(gyr, axis=0)
    features['gyr_max'] = np.max(gyr, axis=0)
    
    # 3. Magnitude (acceleration and gyroscope norms)
    acc_magnitude = np.linalg.norm(acc, axis=1)
    gyr_magnitude = np.linalg.norm(gyr, axis=1)
    
    features['acc_mag_mean'] = np.mean(acc_magnitude)
    features['acc_mag_std'] = np.std(acc_magnitude)
    features['gyr_mag_mean'] = np.mean(gyr_magnitude)
    features['gyr_mag_std'] = np.std(gyr_magnitude)
    
    # 4. Cadence regularity (from acceleration peaks)
    # Find local maxima in acceleration magnitude
    acc_peaks = []
    for i in range(1, len(acc_magnitude) - 1):
        if acc_magnitude[i] > acc_magnitude[i-1] and acc_magnitude[i] > acc_magnitude[i+1]:
            acc_peaks.append(i)
    
    if len(acc_peaks) > 1:
        peak_intervals = np.diff(acc_peaks)
        features['cadence_regularity'] = np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-6)
        features['peak_count'] = len(acc_peaks)
    else:
        features['cadence_regularity'] = 0.0
        features['peak_count'] = 0.0
    
    # 5. Tremor (high-frequency content in gyroscope)
    # Energy in high frequencies
    gyr_fft = np.abs(np.fft.fft(gyr, axis=0))
    high_freq_energy = np.sum(gyr_fft[len(gyr_fft)//2:], axis=0)
    features['tremor_energy'] = np.mean(high_freq_energy)
    
    # 6. Asymmetry (if we have bilateral data)
    # For unilateral sensor: approximate using variation
    features['acc_cv'] = np.std(np.mean(acc, axis=0)) / (np.mean(np.mean(acc, axis=0)) + 1e-6)
    
    # 7. Energy
    features['acc_energy'] = np.sum(acc ** 2)
    features['gyr_energy'] = np.sum(gyr ** 2)
    
    # Flatten to array for sklearn
    feature_array = np.concatenate([
        features['acc_mean'],
        features['acc_std'],
        features['acc_min'],
        features['acc_max'],
        features['gyr_mean'],
        features['gyr_std'],
        features['gyr_min'],
        features['gyr_max'],
        [features['acc_mag_mean']],
        [features['acc_mag_std']],
        [features['gyr_mag_mean']],
        [features['gyr_mag_std']],
        [features['cadence_regularity']],
        [features['peak_count']],
        [features['tremor_energy']],
        [features['acc_cv']],
        [features['acc_energy']],
        [features['gyr_energy']],
    ])
    
    return features, feature_array


def extract_subject_features(subject_id):
    """
    Extract features for a subject (average across all walking trials).
    
    Args:
        subject_id: e.g., 'SUB01'
    
    Returns:
        (features_array, n_files_loaded) or (None, 0) if no files found
    """
    # Find all files for this subject
    pattern = f"{subject_id}_*.csv"
    files = sorted(IMU_DIR.glob(pattern))
    
    if not files:
        return None, 0
    
    all_features = []
    for file_path in files:
        try:
            data = read_gait_file(file_path)
            _, feat_array = extract_gait_features(data)
            all_features.append(feat_array)
        except Exception as e:
            print(f"  ⚠ Failed to load {file_path}: {e}")
            continue
    
    if not all_features:
        return None, len(files)
    
    # Average features across trials
    subject_features = np.mean(all_features, axis=0)
    
    return subject_features, len(files)


# ==================== TRAINING ====================
def train_rf(X, y, groups, n_folds=5):
    """
    Train Random Forest with stratified group k-fold cross-validation.
    
    Returns:
        (clf, cv_results)
    """
    print(f"\nTraining Random Forest with {n_folds}-fold CV...")
    
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    fold_results = []
    models = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subject_ids = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}:")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            random_state=SEED,
            n_jobs=-1,
            class_weight='balanced',
        )
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_val_scaled)
        y_proba = clf.predict_proba(X_val_scaled)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_val, y_proba[:, 1])
        except:
            auc = 0.0
        
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'roc_auc': auc,
        })
        
        print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        models.append((clf, scaler))
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subject_ids.extend(groups[val_idx])
    
    # Summary metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    summary = {
        'model': 'Random Forest',
        'n_folds': n_folds,
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred),
        'recall': recall_score(all_y_true, all_y_pred),
        'roc_auc': roc_auc_score(all_y_true, all_y_proba[:, 1]) if len(np.unique(all_y_true)) > 1 else 0.0,
        'fold_results': fold_results,
        'prediction_level': 'subject_oof',
        'outputs': {
            'predictions_npz': str(OUTPUT_DIR / 'predictions.npz'),
            'models_pkl': str(OUTPUT_DIR / 'rf_models.pkl'),
            'cv_results_csv': str(OUTPUT_DIR / 'cv_results.csv'),
        },
    }
    
    # Save predictions
    np.savez(
        OUTPUT_DIR / 'predictions.npz',
        subject_ids=np.array(all_subject_ids, dtype='U32'),
        y_true=all_y_true,
        y_pred=all_y_pred,
        y_proba=all_y_proba,
    )
    print("\n✓ Predictions saved to predictions.npz")
    
    # Save models
    with open(OUTPUT_DIR / 'rf_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("✓ Models saved to rf_models.pkl")
    
    return summary, all_y_true, all_y_pred, all_y_proba


# ==================== MAIN ====================
def main():
    seed_everything()
    
    print("\n" + "="*60)
    print("RANDOM FOREST BASELINE FOR GAIT")
    print("="*60)
    
    # Load labels
    print("\nLoading PDFE labels...")
    df_labels = pd.read_csv(LABELS_FILE)
    
    severity_map = {}
    for _, row in df_labels.iterrows():
        subject_id = f"SUB{int(row['PDFE'].split('PDFE')[1]):02d}"
        hy_score = row['H Y']
        label = 0 if hy_score <= 2.0 else 1
        severity_map[subject_id] = {
            'label': label,
            'hy_score': hy_score,
            'pdfe_id': row['PDFE'],
        }
    
    print(f"Loaded {len(severity_map)} subjects")
    print(f"  Mild (≤2.0): {sum(1 for v in severity_map.values() if v['label']==0)}")
    print(f"  Moderate/Severe (>2.0): {sum(1 for v in severity_map.values() if v['label']==1)}")
    
    # Extract features for all subjects
    print("\nExtracting gait features...")
    X_list = []
    y_list = []
    groups_list = []
    subject_ids = []
    
    for subject_id in sorted(severity_map.keys()):
        print(f"  {subject_id}...", end=" ", flush=True)
        
        features, n_files = extract_subject_features(subject_id)
        
        if features is None:
            print(f"✗ No data")
            continue
        
        X_list.append(features)
        y_list.append(severity_map[subject_id]['label'])
        groups_list.append(subject_id)
        subject_ids.append(subject_id)
        
        print(f"✓ ({n_files} files)")
    
    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)
    
    print(f"\nFinal dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Train with cross-validation
    summary, y_true, y_pred, y_proba = train_rf(X, y, groups, n_folds=N_FOLDS)
    
    print("\n" + "="*60)
    print("RF SUMMARY")
    print("="*60)
    print(f"Accuracy:  {summary['accuracy']:.4f}")
    print(f"F1 Score:  {summary['f1']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall:    {summary['recall']:.4f}")
    print(f"ROC AUC:   {summary['roc_auc']:.4f}")
    
    # Save results
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(OUTPUT_DIR / 'cv_results.csv', 'w') as f:
        df_results = pd.DataFrame(summary['fold_results'])
        df_results.to_csv(f, index=False)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Random Forest Confusion Matrix')
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Results saved to", OUTPUT_DIR)
    print("  - summary.json")
    print("  - cv_results.csv")
    print("  - predictions.npz")
    print("  - rf_models.pkl")
    print("  - confusion_matrix.png")


if __name__ == '__main__':
    main()
