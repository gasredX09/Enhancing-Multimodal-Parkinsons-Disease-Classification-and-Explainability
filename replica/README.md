# Multimodal Parkinson Project - Replica Notes

This document summarizes what the Multimodal-Parkinson-Disease-Prediction-With-XAI directory implements, and how to reproduce the pipeline with the same data and structure. It is written to mirror what the code actually does (including hard-coded paths and data alignment decisions).

## What the project implements

### 1) Unimodal pipelines (feature extraction)

Speech (EfficientNet-B0 on Mel spectrograms)

- Script: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/UNIMODAL/train_speech.py
- Input data layout (expected):
  - Speech/read_text/HC/*.wav
  - Speech/read_text/PD/*.wav
  - Speech/spontaneous/HC/*.wav
  - Speech/spontaneous/PD/*.wav
- Preprocessing:
  - Resample to 16 kHz, mono
  - Fixed 5-second length with pad or trim
  - Mel spectrogram (n_mels=128, n_fft=1024, hop_length=256)
  - Normalization and 3-channel repeat for EfficientNet input
  - Augmentations via audiomentations (noise, time stretch, pitch shift, gain, shift)
- Model:
  - EfficientNet-B0 pretrained on ImageNet
  - Classifier head replaced for 2 classes
- Outputs (saved in current working directory):
  - best_efficientnet_audio.pth, best_efficientnet_audio.pkl
  - feature_vectors.npy (global average pooled features, 1280-D)
  - labels.npy
  - plots/ training curves

Handwriting (ResNet-50 on spiral/wave images)

- Script: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/UNIMODAL/train_handwriting.py
- Input data layout (expected):
  - Dataset/healthy/*
  - Dataset/parkinson/*
- Preprocessing:
  - Grayscale to 3-channel
  - Resize 224x224
  - Augmentations for train (rotation, affine, flip)
- Model:
  - ResNet-50 pretrained on ImageNet
  - Final layer replaced for 2 classes
  - Optional freezing of early layers
- Outputs:
  - output_resnet50/best_resnet50.pth and .pkl
  - output_resnet50/features.npy and labels.npy
  - plots/ training curves

Gait (autoencoder embeddings + clustering)

- Script: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/UNIMODAL/train_gait.py
- Input data layout (expected):
  - DATA_DIR points to a folder of .txt gait files
  - Filenames like GaCo*, GaPt*, JuCo* are used to infer subject
- Preprocessing:
  - Drop time column
  - Sliding windows of 2 seconds at 100 Hz (200 samples), 50% overlap
  - StandardScaler fit on flattened windows
- Model:
  - TCN autoencoder (requires pretrained weights at outputs/tcn_autoencoder.pt)
  - Encoder embeddings extracted per window
  - KMeans clustering on PCA-reduced embeddings
  - Optional classifier trained on pseudo-labels
- Outputs:
  - outputs_from_saved_ae/embeddings.npy, feature_vectors.npy
  - outputs_from_saved_ae/labels.npy (cluster IDs)
  - outputs_from_saved_ae/scaler.pkl
  - windows_meta.csv, summary_clusters.csv, plots

Important note: bimodal and trimodal scripts expect gait features in output_gait/features.npy with 18 features per subject, but this extraction step is not implemented in this repo. You must either (a) create output_gait/features.npy and labels.npy with shape (subjects, 18) yourself or (b) update paths and code to use outputs_from_saved_ae if you want to fuse autoencoder embeddings.

### 2) Bimodal fusion

Gait + Speech (late fusion, weighted)

- Script: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/BIMODAL/gaitspeech.py
- Inputs:
  - output_EFFICIENTb0_Speech/feature_vectors.npy
  - output_gait/features.npy and labels.npy
- Steps:
  - Aggregate speech features per subject (mean)
  - PCA speech to 50D
  - Train separate XGBoost models (speech and gait)
  - Fuse probabilities via weighted sum and threshold search
- Outputs:
  - latefusion_model.pkl and .pth
  - latefusion_features.npy, latefusion_labels.npy

Handwriting + Gait (early fusion)

- Script: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/BIMODAL/hwgait.py
- Inputs:
  - output_resnet50/features.npy and labels.npy
  - output_gait/features.npy
- Steps:
  - Aggregate handwriting features per subject (mean)
  - Concatenate with gait features (20D)
  - Standardize and add Gaussian noise
  - Train small XGBoost, threshold=0.7
- Outputs:
  - BIMODAL/features.npy, BIMODAL/labels.npy
  - xgb_bimodal_model.pkl and .pth

Handwriting + Speech (early fusion with SMOTE)

- Script: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/BIMODAL/hwspeech.py
- Inputs:
  - output_EFFICIENTb0_Speech/feature_vectors.npy
  - output_resnet50/features.npy and labels.npy
- Steps:
  - Concatenate audio and image features
  - Standardize
  - SMOTE for class balance
  - XGBoost, threshold=0.45
- Outputs:
  - bimodal_model.pkl and .pth
  - bimodal_features.npy, bimodal_labels.npy

### 3) Trimodal fusion (speech + gait + handwriting)

- Script: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/TRIMODAL/trimodal.py
- Inputs:
  - output_EFFICIENTb0_Speech/feature_vectors.npy
  - output_gait/features.npy
  - output_resnet50/features.npy and labels.npy
- Steps (as coded):
  - Use gait subjects count (306) as alignment reference
  - Aggregate speech features per subject (mean)
  - PCA speech to 50D
  - Handwriting: stratified sample to 306 and PCA to 2D
  - Concatenate [speech(50) + gait(18) + handwriting(2)] to 70D
  - Train/test split (60/40), StandardScaler, SMOTE
  - XGBoost classifier
  - Save features, labels, model, SHAP summaries, learning curve, PCA/t-SNE/UMAP plots
- Outputs:
  - TRIMODAL/features.npy, labels.npy
  - xgb_trimodal_model.pkl and .pth
  - speech_pca.pkl, handwriting_pca.pkl
  - SHAP plots and embedding plots

### 4) Inference and XAI

- Single sample inference: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/TRIMODAL/prediction.py
  - Extracts speech features (EfficientNet), handwriting features (ResNet), gait features (from .txt using scaler)
  - Applies PCA and concatenates to 70D
  - Uses trained XGBoost to predict

- Streamlit demo and XAI: ../../Multimodal-Parkinson-Disease-Prediction-With-XAI/TRIMODAL/dashboard.py
  - Upload speech, handwriting, gait
  - Predict and visualize:
    - Grad-CAM for handwriting
    - Grad-CAM++ for speech
    - SHAP for gait features

- Additional utilities:
  - mel.py: save mel-spectrogram image for a wav file
  - grad-speech.py: batch Grad-CAM++ for speech
  - grad-hw.py: batch Grad-CAM for handwriting
  - tsne.py: t-SNE and UMAP plots for multiple feature sets
  - assets/vis.py and assets/vis2.py: correlation and performance plots

## How to replicate with the same data

### 0) Download and place datasets

Use the same sources referenced in the project README:

- Speech: Figshare voice samples
- Gait: Figshare gait turning-in-place dataset
- Handwriting: Kaggle Parkinsons drawings

Recommended local layout (matches code defaults):

- Multimodal-Parkinson-Disease-Prediction-With-XAI/Speech/read_text/HC
- Multimodal-Parkinson-Disease-Prediction-With-XAI/Speech/read_text/PD
- Multimodal-Parkinson-Disease-Prediction-With-XAI/Speech/spontaneous/HC
- Multimodal-Parkinson-Disease-Prediction-With-XAI/Speech/spontaneous/PD
- Multimodal-Parkinson-Disease-Prediction-With-XAI/Dataset/healthy
- Multimodal-Parkinson-Disease-Prediction-With-XAI/Dataset/parkinson
- Gait data folder for .txt files (update DATA_DIR in train_gait.py)

### 1) Create and activate a Python environment

The scripts use the following packages (minimum set):

- numpy, pandas, scipy, matplotlib, seaborn
- torch, torchvision, torchaudio
- scikit-learn, imbalanced-learn, joblib
- xgboost, shap, umap-learn
- audiomentations, captum, torchcam
- opencv-python, pillow
- streamlit (for dashboard)

### 2) Train unimodal models and extract features

Speech:

- Run train_speech.py from a working directory where you want outputs saved.
- If you want the same paths as other scripts, run it inside output_EFFICIENTb0_Speech or move the outputs there after training.

Handwriting:

- Run train_handwriting.py from the project root.
- Output goes to output_resnet50/ and plots/.

Gait:

- Ensure outputs/tcn_autoencoder.pt exists. This repo does not train the autoencoder; you must supply pretrained weights.
- Update DATA_DIR in train_gait.py to your gait folder.
- Run train_gait.py to generate outputs_from_saved_ae/ and scaler.pkl.

### 3) Prepare gait features for fusion (required)

Bimodal and trimodal scripts expect:

- output_gait/features.npy with shape (subjects, 18)
- output_gait/labels.npy with shape (subjects,)

This extraction is not included in the repo. To replicate the fusion code exactly, you must either:

- Produce 18D gait feature vectors per subject (using your own feature engineering), or
- Modify the fusion scripts to use outputs_from_saved_ae/feature_vectors.npy and align dimensions accordingly.

### 4) Run bimodal fusion

- Run gaitspeech.py, hwspeech.py, hwgait.py after all unimodal features exist.
- These scripts save their models and evaluation outputs into the BIMODAL folder.

### 5) Run trimodal fusion

- Run TRIMODAL/trimodal.py after speech, handwriting, and gait features are ready.
- This saves the PCA models, XGBoost model, and explainability plots into TRIMODAL.

### 6) Run inference and dashboard (optional)

- Update paths in TRIMODAL/prediction.py and TRIMODAL/dashboard.py to your local outputs.
- For a demo UI, run dashboard.py with streamlit:
  - streamlit run TRIMODAL/dashboard.py

## Known path and alignment assumptions in code

- Most scripts contain hard-coded Windows paths (D:\...). These must be updated for macOS.
- Speech features are averaged per subject based on simple integer division of sample count.
- Handwriting is sampled to match gait subject count (306) using stratified sampling.
- Trimodal fusion assumes 50 speech PCA features, 18 gait features, and 2 handwriting PCA features (total 70).

## Suggested replication checklist

- Verify dataset layouts match the expected folder structures.
- Update all hard-coded paths to local macOS paths.
- Generate unimodal features and confirm shapes.
- Create or replace gait feature extraction so output_gait/features.npy exists.
- Run bimodal scripts; verify saved models and report metrics.
- Run trimodal script; verify SHAP and visualization outputs.
- Run prediction.py and dashboard.py to validate end-to-end inference.
