# Gait-Based Parkinson's Disease Severity Classification

This module implements unimodal classification of Parkinson's Disease **severity** using gait IMU sensor data.

## Classification Task

**Binary severity classification:**
- **Class 0**: Mild PD (Hoehn & Yahr score ≤ 2.0)  
- **Class 1**: Moderate/Severe PD (H&Y score > 2.0)

**Note:** The figshare dataset contains only PD patients (no healthy controls), so this is a severity stratification task rather than HC vs PD detection.

## Model Architecture

**Temporal Convolutional Network (TCN)**
- 4-layer TCN with increasing dilation rates (1, 2, 4, 8)
- Channel progression: input → 64 → 64 → 128 → 128
- Residual connections for better gradient flow
- Global average pooling + 3-layer classifier head

**Advantages over the original:**
1. **Supervised learning** instead of unsupervised clustering
2. **Proper stratified group k-fold CV** to prevent data leakage
3. **Modern training practices**: learning rate scheduling, early stopping, gradient clipping
4. **Better metrics tracking**: precision, recall, F1, ROC AUC

## Data Requirements

- **Input**: IMU sensor data (.txt files preferred, .csv contains Excel format) in `data/gait/figshare/IMU/`
- **Labels**: PDFEinfo.csv with H&Y scores for severity stratification
- **Format**: Time-series data with multiple sensor channels (ACC ML/AP/SI, GYR ML/AP/SI)

## Running

```bash
cd /ocean/projects/med260006p/shared/biomedAI/project/replica/02_unimodal/03_gait
python train_gait.py
```

## Configuration

Edit constants at the top of `train_gait.py`:

```python
# Data parameters
WINDOW_SEC = 2.0          # Window size in seconds
WINDOW_STEP = WINDOW_SAMPLES // 2  # 50% overlap

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15             # Early stopping patience
LR = 1e-3
N_FOLDS = 5              # Cross-validation folds

# Model parameters
TCN_CHANNELS = [64, 64, 128, 128]
DILATIONS = [1, 2, 4, 8]
DROPOUT = 0.3
```

## Outputs

All outputs saved to `outputs/unimodal_gait/`:

```
outputs/unimodal_gait/
├── scaler.pkl                    # Fitted StandardScaler
├── cv_results.csv                # Per-fold metrics
├── summary.json                  # Overall summary
└── fold_1/
    ├── best_model.pth            # Best model checkpoint
    ├── confusion_matrix.png      # Confusion matrix plot
    └── classification_report.txt # Detailed metrics
```

## Expected Performance

Based on literature for gait-based PD severity classification:
- **Accuracy**: 70-85% (severity stratification is harder than HC vs PD)
- **F1 Score**: 0.68-0.82
- **ROC AUC**: 0.75-0.88

Note: Performance depends on data quality, sensor placement, walking conditions, and class imbalance (6 mild vs 29 moderate/severe patients).

## Improvements Over Original

1. ✅ **Supervised learning** with real clinical labels (H&Y scores, not pseudo-labels from clustering)
2. ✅ **Stratified group k-fold CV** prevents subject leakage between train/val
3. ✅ **Modern PyTorch practices**: DataLoader, learning rate scheduling, early stopping
4. ✅ **Comprehensive metrics**: precision, recall, F1, AUC, confusion matrices
5. ✅ **Clean, readable code** with proper documentation
6. ✅ **No hardcoded Windows paths** - uses pathlib for cross-platform compatibility
7. ✅ **Clinically meaningful task**: Severity stratification using validated H&Y scale

## Citation

If you use this code, please cite:
- Original dataset: [Gait in Parkinson's Disease Dataset](https://physionet.org/content/gaitpdb/1.0.0/)
- TCN architecture: Bai et al., "Temporal Convolutional Networks" (2018)
