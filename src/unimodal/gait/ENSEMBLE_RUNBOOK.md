# Gait Ensemble Training Strategy
## Complete Plan + Runbook

---

## Overview

You will train **3 independent gait models** and then **fuse their predictions** to achieve better performance than any single model.

### The 3 Tasks

| Task | Dataset | Model | Input Features | Target | Expected Accuracy |
|------|---------|-------|-----------------|--------|-------------------|
| **Task 1: PDFE Severity** | figshare (35 subjects) | TCN | 6 IMU channels | Mild vs Moderate/Severe | 70-80% |
| **Task 2: WearGait Embeddings** | weargait (100+ subjects) | TCN | 24 sensor channels | HC vs PD | 75-85% |
| **Task 3: RF Baseline** | figshare (35 subjects) | Random Forest | Engineered gait features | Mild vs Moderate/Severe | 65-75% |

### Why This Works Better Than Combined Training

- **Task 1** learns severity patterns from PDFE data (specialized signal)
- **Task 2** learns general HC/PD distinction from wearable data (cleaner, more subjects)
- **Task 3** captures hand-engineered clinical features as a baseline
- **Fusion** learns which task is most reliable for predictions (weighted ensemble)

---

## Step 1: Preparation (15 min)

Ensure data is in place:

```bash
ls -lh data/gait/figshare/IMU/         # ~200 files, figshare data
ls -lh data/gait/weargait/            # Wearable dataset 
ls -lh outputs/unimodal_gait/          # Output dir (will be created)
```

Verify Python environment:

```bash
conda activate biomedai
pip install -q torch scikit-learn pandas numpy matplotlib seaborn tqdm
```

---

## Step 2: Train All 3 Tasks (1-2 hours, GPU-accelerated)

### Option A: Train all at once

```bash
cd src/unimodal/gait

# Train all 3 tasks with default config
python gait_ensemble_orchestrator.py --tasks all --device cuda

# Expected output:
# ✓ PDFE_Severity_Classification/ → outputs/
# ✓ WearGait_Task_Embeddings/ → outputs/
# ✓ Random_Forest_Baseline/ → outputs/
```

### Option B: Train individually (if you want to debug)

```bash
# Task 1: PDFE severity (supervised learning on figshare)
python gait_ensemble_orchestrator.py --tasks pdfe --device cuda --n-folds 5

# Task 2: WearGait embeddings (from wearable sensors)
python gait_ensemble_orchestrator.py --tasks weargait --device cuda --n-folds 5

# Task 3: RF baseline (engineered features)
python gait_ensemble_orchestrator.py --tasks rf --device cpu
```

### What Each Task Does

#### Task 1: PDFE Severity Classification

**Implementation location:** `train_gait.py` (already exists, will be integrated)

Steps:
1. Load figshare IMU files (SUB01-SUB35)
2. Map subject IDs to H&Y severity labels → mild (≤2.0) vs moderate/severe (>2.0)
3. Extract sliding windows (2 sec, 50% overlap)
4. Train TCN with stratified group k-fold CV (n_splits=5)
5. Save:
   - `outputs/unimodal_gait/PDFE_Severity_Classification/cv_results.csv`
   - `outputs/unimodal_gait/PDFE_Severity_Classification/predictions.npz`

**Expected metrics:** Accuracy 70-80%, F1 0.68-0.78

---

#### Task 2: WearGait Embeddings

**Implementation location:** `train_weargait_embeddings.py` (already exists, will be integrated)

Steps:
1. Load wearable gait data (multi-sensor: lower back, ankles, CoP)
2. Filter to HC vs PD classification (not severity)
3. Parse task labels (SelfPace, HurriedPace, TUG)
4. Train TCN to extract embeddings
5. Classify based on embeddings
6. Save embeddings for fusion

**Expected metrics:** Accuracy 75-85%, F1 0.73-0.83

---

#### Task 3: RF Baseline

**Implementation location:** `train_gait_rf.py` (will create)

Steps:
1. Extract engineered features from PDFE data:
   - Stride length variability (standard deviation)
   - Cadence regularity (coefficient of variation)
   - Bilateral asymmetry (left vs right diff)
   - Power spectral density peaks
2. Train Random Forest classifier
3. Report feature importance

**Expected metrics:** Accuracy 65-75%, F1 0.60-0.72

---

## Step 3: Evaluate Individual Tasks (5 min)

Check individual performance:

```bash
# View PDFE results
cat outputs/unimodal_gait/PDFE_Severity_Classification/summary.json

# View WearGait results
cat outputs/unimodal_gait/WearGait_Task_Embeddings/summary.json

# View RF results
cat outputs/unimodal_gait/Random_Forest_Baseline/summary.json
```

**Key question:** Which task is strongest? (You'll use this in fusion to assign weights)

---

## Step 4: Fuse Predictions (20 min)

Run fusion with all 4 strategies:

```bash
cd src/unimodal/gait

python ensemble_fusion.py --strategy all --seed 42
```

**This will:**
1. Load predictions from all 3 tasks
2. Test 4 fusion strategies:
   - **Weighted Average:** w₁·p₁ + w₂·p₂ + w₃·p₃ (weights learned on val set)
   - **Stacking:** Use task predictions as features for meta-learner (LogisticRegression)
   - **Voting:** Majority vote from 3 tasks
   - **Calibrated:** Late fusion with Platt scaling

**Output:** `outputs/unimodal_gait/fusion_results/`
- `fusion_comparison.json` → metrics for each strategy
- `fusion_comparison.png` → bar plot of accuracy/F1/AUC

---

## Step 5: Compare All Results (5 min)

Create summary report:

```bash
# View fusion results
cat outputs/unimodal_gait/fusion_results/fusion_comparison.json

# Expected output:
# {
#   "weighted": {"accuracy": 0.82, "f1": 0.79, "roc_auc": 0.85},
#   "stacking": {"accuracy": 0.85, "f1": 0.82, "roc_auc": 0.88},
#   "voting": {"accuracy": 0.80, "f1": 0.77, "roc_auc": 0.83},
#   "best_strategy": "stacking"
# }
```

**Goal:** Ensemble (stacking/weighted) should beat any single task by 2-5%.

---

## Expected Performance Summary

### Individual Tasks

| Task | Accuracy | F1 | AUC |
|------|----------|-----|-----|
| PDFE TCN | 73% | 0.71 | 0.78 |
| WearGait TCN | 81% | 0.79 | 0.86 |
| RF Baseline | 70% | 0.68 | 0.76 |

### Fusion Strategies

| Fusion Strategy | Accuracy | F1 | AUC | Remark |
|-----------------|----------|-----|-----|--------|
| Weighted Avg | 80% | 0.78 | 0.84 | Simple, interpretable |
| **Stacking** | **84%** | **0.82** | **0.88** | Best performance |
| Voting | 79% | 0.77 | 0.82 | Conservative |
| Calibrated | 82% | 0.80 | 0.86 | Robust |

**Expected gain:** ~11% over worst baseline (RF), ~3% over best single task (WearGait)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

```bash
conda activate biomedai
pip install torch torchvision torchaudio
```

### Issue: "data/gait/figshare/IMU/ not found"

Check that data was downloaded:
```bash
ls -la data/gait/figshare/IMU/ | head -5
# Should show: SUB01_1.csv, SUB01_2.csv, etc.
```

If missing, download from figshare link in your data documentation.

### Issue: "GPU memory error"

Reduce batch size and epochs:
```bash
# Edit gait_ensemble_orchestrator.py:
TASK_CONFIGS['pdfe'].batch_size = 16
TASK_CONFIGS['pdfe'].epochs = 50
```

### Issue: "Predictions shape mismatch in fusion"

Check that all tasks saved predictions in same format:
```bash
# Verify prediction files exist
ls -lh outputs/unimodal_gait/*/predictions.npz
```

If missing, re-run individual task training with `--save-predictions` flag.

---

## Next Steps (After Ensemble)

1. **Explainability:** Which features did each task learn?
   - Use SHAP for feature importance
   - Attention visualization for TCN

2. **Ablation Study:** Remove each task, measure accuracy drop
   - If PDFE drops accuracy by 1%, it's not helping → drop it
   - If WearGait drops it by 5%, it's critical → keep it

3. **Domain Validation:** Do learned features match clinical knowledge?
   - PDFE: Should detect H&Y severity patterns
   - WearGait: Should detect HC vs PD stride differences
   - RF: Should weight stride irregularity heavily

4. **Cross-validation:** Ensure stratified group k-fold to prevent subject leakage

---

## File Structure After Training

```
outputs/unimodal_gait/
├── PDFE_Severity_Classification/
│   ├── cv_results.csv
│   ├── summary.json
│   ├── fold_1/
│   │   ├── best_model.pth
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   └── fold_2/ ... fold_5/
│
├── WearGait_Task_Embeddings/
│   ├── embeddings.npz
│   ├── cv_results.csv
│   ├── summary.json
│   └── fold_1/ ... fold_5/
│
├── Random_Forest_Baseline/
│   ├── feature_importance.json
│   ├── summary.json
│   └── cv_results.csv
│
└── fusion_results/
    ├── fusion_comparison.json
    ├── fusion_comparison.png
    └── best_fusion_model.pkl
```

---

## Key Code Locations

- **Orchestrator:** `src/unimodal/gait/gait_ensemble_orchestrator.py`
- **Task 1 impl:** `src/unimodal/gait/train_gait.py` (integrate)
- **Task 2 impl:** `src/unimodal/gait/train_weargait_embeddings.py` (integrate)
- **Task 3 impl:** `src/unimodal/gait/train_gait_rf.py` (new file)
- **Fusion:** `src/unimodal/gait/ensemble_fusion.py`

---

## Quick Start (Copy-Paste)

```bash
cd /ocean/projects/med260006p/shared/biomedAI/project

# Prep
conda activate biomedai

# Train all tasks
cd src/unimodal/gait
python gait_ensemble_orchestrator.py --tasks all --device cuda

# Fuse predictions
python ensemble_fusion.py --strategy all

# Check results
cat ../../../outputs/unimodal_gait/fusion_results/fusion_comparison.json
```

---

## Questions?

- **Task 1 failing?** Check `train_gait.py` for dataset loading issues
- **Task 2 failing?** Check `train_weargait_embeddings.py` for channel parsing
- **Task 3 failing?** Check feature extraction logic in `train_gait_rf.py`
- **Fusion bad?** Ensure all task predictions have same shape (n_samples, 2)
