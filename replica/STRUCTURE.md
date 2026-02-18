# Project Structure - Multimodal Parkinson Prediction Replica

This document outlines the organized folder structure for replicating the multimodal Parkinson's Disease prediction pipeline.

```
/Users/aryansharanreddyguda/biomedAI/project/replica/
│
├── README.md                          # Main project documentation (replication guide)
│
├── 01_eda/                            # Exploratory Data Analysis
│   ├── gait_eda.ipynb                 # Gait data EDA (time series, signals, statistics)
│   ├── speech_eda.ipynb               # Speech data EDA (spectrograms, audio features)
│   ├── handwriting_eda.ipynb          # Handwriting data EDA (image stats, distributions)
│   └── multimodal_alignment.ipynb     # Check alignment across modalities
│
├── 02_unimodal/                       # Unimodal Feature Extraction & Models
│   ├── 01_speech/
│   │   ├── train_speech.py            # EfficientNet-B0 on Mel spectrograms
│   │   ├── speech_features.ipynb      # Training & evaluation notebook
│   │   └── config_speech.yaml         # Hyperparameters
│   ├── 02_handwriting/
│   │   ├── train_handwriting.py       # ResNet-50 on spiral/wave images
│   │   ├── handwriting_features.ipynb # Training & evaluation notebook
│   │   └── config_handwriting.yaml    # Hyperparameters
│   ├── 03_gait/
│   │   ├── train_gait.py              # Autoencoder embeddings + clustering
│   │   ├── gait_features.ipynb        # Feature extraction notebook
│   │   └── config_gait.yaml           # Hyperparameters
│   └── utils.py                       # Shared utilities (preprocessing, augmentation)
│
├── 03_bimodal/                        # Bimodal Fusion Models
│   ├── 01_speech_gait/
│   │   ├── train_fusion.py            # Late fusion (weighted probability)
│   │   └── evaluate.ipynb             # Evaluation & SHAP
│   ├── 02_handwriting_gait/
│   │   ├── train_fusion.py            # Early fusion
│   │   └── evaluate.ipynb             # Evaluation & SHAP
│   ├── 03_handwriting_speech/
│   │   ├── train_fusion.py            # Early fusion with SMOTE
│   │   └── evaluate.ipynb             # Evaluation & SHAP
│   └── config_bimodal.yaml            # Common bimodal params
│
├── 04_trimodal/                       # Trimodal Fusion Model
│   ├── train_trimodal.py              # Trimodal early fusion + XGBoost
│   ├── trimodal_fusion.ipynb          # Training & evaluation notebook
│   ├── config_trimodal.yaml           # Hyperparameters
│   └── analysis.ipynb                 # Explainability (SHAP, Grad-CAM, t-SNE, UMAP)
│
├── 05_inference/                      # Inference & Deployment
│   ├── predict_sample.py              # Single sample prediction
│   ├── predict_batch.py               # Batch prediction
│   ├── streamlit_dashboard.py         # Streamlit interactive UI
│   ├── xai_visualizations.py          # Grad-CAM, Grad-CAM++, SHAP
│   └── config_inference.yaml          # Inference config
│
├── data/                              # Organized data structure (local copies/links)
│   ├── Speech/
│   │   ├── read_text/
│   │   │   ├── HC/
│   │   │   └── PD/
│   │   └── spontaneous/
│   │       ├── HC/
│   │       └── PD/
│   ├── Handwriting/
│   │   ├── healthy/
│   │   └── parkinson/
│   ├── Gait/
│   │   └── (raw .txt files)
│   └── README_data.md                 # Data download & setup instructions
│
├── outputs/                           # All model outputs & results
│   ├── gait_eda/                      # Gait EDA results
│   ├── speech_eda/                    # Speech EDA results
│   ├── handwriting_eda/               # Handwriting EDA results
│   ├── unimodal_speech/               # Speech model & features
│   │   ├── best_model.pth
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   └── plots/
│   ├── unimodal_handwriting/          # Handwriting model & features
│   │   ├── best_model.pth
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   └── plots/
│   ├── unimodal_gait/                 # Gait embeddings & features
│   │   ├── embeddings.npy
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   ├── scaler.pkl
│   │   └── plots/
│   ├── bimodal_sg/                    # Speech + Gait fusion
│   │   ├── model.pkl
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   └── metrics.json
│   ├── bimodal_hg/                    # Handwriting + Gait fusion
│   │   ├── model.pkl
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   └── metrics.json
│   ├── bimodal_hs/                    # Handwriting + Speech fusion
│   │   ├── model.pkl
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   └── metrics.json
│   ├── trimodal/                      # Trimodal model
│   │   ├── xgb_model.pkl
│   │   ├── speech_pca.pkl
│   │   ├── handwriting_pca.pkl
│   │   ├── gait_scaler.pkl
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   ├── shap_plots/
│   │   ├── gradcam_plots/
│   │   ├── embedding_plots/
│   │   ├── confusion_matrix.png
│   │   └── metrics.json
│   └── inference/                     # Demo predictions & visualizations
│       ├── sample_predictions.csv
│       ├── xai_explanations/
│       └── dashboard_cache/
│
├── configs/                           # Centralized configuration files
│   ├── default.yaml                   # Default settings
│   ├── paths.yaml                     # Path mappings (for macOS)
│   ├── hyperparams.yaml               # ML hyperparameters
│   └── preprocess.yaml                # Preprocessing settings
│
├── scripts/                           # Utility scripts
│   ├── setup_env.sh                   # Environment setup script
│   ├── download_data.py               # Download datasets (if available)
│   ├── validate_data.py               # Data integrity checks
│   ├── preprocess_all.py              # Run all preprocessing
│   └── run_pipeline.py                # End-to-end pipeline
│
└── logs/                              # Training logs & checkpoints
    ├── train_speech.log
    ├── train_handwriting.log
    ├── train_gait.log
    ├── bimodal_*.log
    └── trimodal.log
```

## Directory Purposes

### `01_eda/`

Contains Jupyter notebooks for exploratory data analysis of each modality. Start here to understand data characteristics, distributions, and preprocessing needs.

### `02_unimodal/`

Individual feature extraction and model training for each modality:

- **Speech**: EfficientNet-B0 trained on Mel-spectrograms
- **Handwriting**: ResNet-50 trained on spiral/wave images
- **Gait**: Autoencoder embeddings + clustering from time series

### `03_bimodal/`

Fusion models combining pairs of modalities:

- **Speech + Gait**: Late fusion with probability weighting
- **Handwriting + Gait**: Early fusion with concatenation
- **Handwriting + Speech**: Early fusion with SMOTE balancing

### `04_trimodal/`

Final multimodal fusion combining all three modalities with XGBoost classifier and full explainability analysis.

### `05_inference/`

Scripts for deploying the trained model:

- Single/batch predictions
- Streamlit interactive dashboard
- XAI visualizations (Grad-CAM, SHAP)

### `data/`

Organized data directory structure. Symlink or copy datasets here to keep things organized. Mirrors the expected layout from original project.

### `outputs/`

All results, models, and artifacts are saved here organized by stage. This prevents cluttering the repo and makes results reproducible.

### `configs/`

Centralized YAML configuration files for paths, hyperparameters, and preprocessing settings. Makes it easy to adapt to different machines/environments.

### `scripts/`

Utility scripts for setup, validation, and running the full pipeline end-to-end.

### `logs/`

Training logs and checkpoint files for debugging and tracking experiments.

## How to Use

1. **Download data** and organize in `data/` subfolder
2. **Run EDA** notebooks in `01_eda/` to understand your data
3. **Train unimodal models** using scripts in `02_unimodal/`
4. **Train bimodal models** using scripts in `03_bimodal/`
5. **Train trimodal model** using scripts in `04_trimodal/`
6. **Run inference** using scripts in `05_inference/`

All outputs are saved to `outputs/` for easy tracking and reproducibility.

---

**Created**: 2026-02-18
**Purpose**: Organized replication of multimodal Parkinson's disease prediction project
