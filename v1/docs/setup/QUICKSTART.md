# Quick Start Guide - Project Setup

Get the multimodal Parkinson prediction replica up and running.

## Step 1: Project Structure

✅ **Already created** organized folders:

```
/Users/aryansharanreddyguda/biomedAI/project/
├── 01_eda/              # EDA notebooks
├── 02_unimodal/         # Unimodal models
├── 03_bimodal/          # Bimodal fusion
├── 04_trimodal/         # Trimodal fusion
├── 05_inference/        # Prediction & deployment
├── data/                # Data directory (organize datasets here)
├── outputs/             # All results & models
├── README.md            # Main documentation
└── STRUCTURE.md         # Detailed structure explanation
```

See [STRUCTURE.md](STRUCTURE.md) for full details.

## Step 2: Environment Setup

### Create conda environment

```bash
cd /Users/aryansharanreddyguda/biomedAI/project

# Create env
conda create -n biomedai python=3.10

# Activate
conda activate biomedai
```

### Install dependencies

```bash
pip install --upgrade pip

# Core data science
pip install numpy pandas scipy matplotlib seaborn scikit-learn

# Deep learning
pip install torch torchvision torchaudio

# Models & ML
pip install xgboost imbalanced-learn joblib

# Audio processing
pip install librosa torchaudio audiomentations soundfile

# Image processing
pip install opencv-python pillow

# Explainability
pip install shap captum umap-learn tensorboard

# For interactive notebooks
pip install jupyter jupyterlab ipywidgets

# Optional: for Streamlit deployment
pip install streamlit

# Optional: for improved notebooks
pip install plotly bokeh
```

## Step 3: Prepare Data

### Option A: Use existing data

If you already have the data, organize it in `/Users/aryansharanreddyguda/biomedAI/project/data/`:

```
data/
├── Speech/
│   ├── read_text/
│   │   ├── HC/          # Healthy control .wav files
│   │   └── PD/          # Parkinson .wav files
│   └── spontaneous/
│       ├── HC/
│       └── PD/
├── Handwriting/
│   ├── healthy/         # Healthy spiral/wave images
│   └── parkinson/       # Parkinson spiral/wave images
└── Gait/
    └── (all .txt gait files)
```

### Option B: Download datasets

The original project uses public datasets:

- **Speech**: <https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127>
- **Gait**: <https://figshare.com/articles/dataset/A_public_dataset_of_video_acceleration_and_angular_velocity_in_individuals_with_Parkinson_s_disease_during_the_turning-in-place_task/14984667>
- **Handwriting**: <https://www.kaggle.com/datasets/kmader/parkinsons-drawings/data>

## Step 4: Run EDA

Start with gait data EDA:

```bash
cd /Users/aryansharanreddyguda/biomedAI/project/notebooks/eda

# Open notebook
jupyter lab gait_eda.ipynb

# OR
jupyter notebook gait_eda.ipynb
```

**What to expect:**

- Loads gait .txt files
- Analyzes signal statistics
- Creates visualizations in `outputs/gait_eda/`
- Generates recommendations for preprocessing

## Step 5: Next Steps (in order)

### Phase 1: Exploratory Analysis

```
01_eda/
├── gait_eda.ipynb           ← Start here
├── speech_eda.ipynb         ← Then speech
├── handwriting_eda.ipynb    ← Then images
└── multimodal_alignment.ipynb ← Alignment check
```

### Phase 2: Unimodal Models

```
02_unimodal/
├── 01_speech/train_speech.py         → EfficientNet-B0
├── 02_handwriting/train_handwriting.py → ResNet-50
└── 03_gait/train_gait.py             → Autoencoder
```

### Phase 3: Bimodal Fusion

```
03_bimodal/
├── 01_speech_gait/
├── 02_handwriting_gait/
└── 03_handwriting_speech/
```

### Phase 4: Trimodal Fusion

```
04_trimodal/train_trimodal.py
```

### Phase 5: Deployment

```
05_inference/
├── predict_sample.py
├── predict_batch.py
└── streamlit_dashboard.py
```

## Step 6: Check Installation

Test that everything is set up correctly:

```python
# test_setup.py
import numpy as np
import pandas as pd
import torch
import torchvision
import sklearn
import xgboost
import shap

print("✅ All packages imported successfully!")
print(f"   NumPy: {np.__version__}")
print(f"   Pandas: {pd.__version__}")
print(f"   PyTorch: {torch.__version__}")
print(f"   Scikit-learn: {sklearn.__version__}")
print(f"   XGBoost: {xgboost.__version__}")
```

Run it:

```bash
python test_setup.py
```

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution**: Make sure the conda environment is activated and all packages are installed

```bash
conda activate biomedai
pip list | grep numpy  # Check if installed
```

### Issue: "No module named jupyter"

**Solution**: Install jupyter

```bash
pip install jupyterlab
```

### Issue: Notebook kernel not found

**Solution**: Install ipykernel in the conda environment

```bash
pip install ipykernel
python -m ipykernel install --user --name biomedai --display-name "biomedai"
```

### Issue: CUDA/GPU not found

**Solution**: This is optional. CPU will work fine, just slower.

```bash
# To enable GPU (NVIDIA only):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory

**Solution**: Reduce batch sizes or process data in chunks. See section on data splitting.

## File Organization Tips

### Keep outputs organized

All model outputs go to `outputs/`:

- `gait_eda/` - Gait analysis results
- `unimodal_speech/` - Speech model & features
- `bimodal_sg/` - Speech+Gait fusion results
- `trimodal/` - Final model & explanations

### Use consistent naming

- Models: `best_model.pth` or `model.pkl`
- Features: `features.npy` (shape: num_samples × num_features)
- Labels: `labels.npy` (shape: num_samples,)

### Create symlinks for data (optional)

If data is elsewhere:

```bash
cd /Users/aryansharanreddyguda/biomedAI/project/data
ln -s /path/to/gait Gait
ln -s /path/to/speech Speech
ln -s /path/to/handwriting Handwriting
```

## Configuration Management

Create a `config.yaml` in project root for paths:

```yaml
# config.yaml
paths:
  gait_data: /Users/aryansharanreddyguda/biomedAI/project/data/Gait
  speech_data: /Users/aryansharanreddyguda/biomedAI/project/data/Speech
  handwriting_data: /Users/aryansharanreddyguda/biomedAI/project/data/Handwriting
  outputs: /Users/aryansharanreddyguda/biomedAI/project/outputs

training:
  speech_epochs: 30
  handwriting_epochs: 50
  batch_size: 16
  learning_rate: 1e-4

preprocessing:
  sample_rate: 100  # Hz for gait
  window_size: 200  # 2 seconds at 100Hz
  window_step: 100  # 50% overlap
```

Then load in Python:

```python
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
print(config['paths']['outputs'])
```

## Next Resources

- **Original project**: `/Users/aryansharanreddyguda/biomedAI/Multimodal-Parkinson-Disease-Prediction-With-XAI/`
- **README**: [../README.md](README.md) - Detailed replication notes
- **Structure**: [STRUCTURE.md](STRUCTURE.md) - Folder organization explanation
- **EDA Guide**: [01_eda/README.md](01_eda/README.md) - How to run EDA notebook

---

**Ready to go!** 🚀

Start with:

```bash
cd /Users/aryansharanreddyguda/biomedAI/project/notebooks/eda
jupyter lab gait_eda.ipynb
```
