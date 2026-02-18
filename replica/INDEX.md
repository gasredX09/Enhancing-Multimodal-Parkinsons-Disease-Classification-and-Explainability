# Multimodal Parkinson Disease Prediction - Project Index

## Quick Navigation

### Documentation (Read First)

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Start here
  - What is created
  - 3-step quick start
  - Next actions

- **[QUICKSTART.md](QUICKSTART.md)** - Detailed setup guide
  - Conda environment setup
  - Package installation
  - Data preparation
  - Running individual components

- **[README.md](README.md)** - Project replication guide
  - Original project summary
  - Architecture and methodology
  - Data sources and citations
  - Assumptions and pitfalls
  - Replication checklist

- **[STRUCTURE.md](STRUCTURE.md)** - Folder organization
  - Directory breakdown
  - Purpose of each section
  - Output locations
  - Organization conventions

### Exploratory Data Analysis

Directory: `01_eda/`

- **[01_eda/gait_eda.ipynb](01_eda/gait_eda.ipynb)** - Ready to use
  - Load and analyze gait time series
  - Signal statistics and distributions
  - Time series visualization
  - FFT frequency analysis
  - Quality checks (NaN, Inf, outliers)
  - Preprocessing recommendations
  - Run: `jupyter lab 01_eda/gait_eda.ipynb`

- **01_eda/README.md** - EDA guide
  - How to run the notebook
  - Expected file formats
  - Output descriptions
  - Troubleshooting

- **To create next**:
  - `speech_eda.ipynb` - Audio data analysis
  - `handwriting_eda.ipynb` - Image data analysis
  - `multimodal_alignment.ipynb` - Alignment check

### Unimodal Models

Directory: `02_unimodal/`

Status: Folder created, scripts pending

To be added:

- `01_speech/` - EfficientNet-B0 on mel spectrograms
- `02_handwriting/` - ResNet-50 on spiral or wave images
- `03_gait/` - Autoencoder embeddings and clustering
- `utils.py` - Shared preprocessing functions

### Bimodal Fusion

Directory: `03_bimodal/`

Status: Folder created, models pending

To be added:

- `01_speech_gait/` - Late fusion of speech and gait
- `02_handwriting_gait/` - Early fusion of handwriting and gait
- `03_handwriting_speech/` - Early fusion with SMOTE
- `config_bimodal.yaml` - Shared parameters

### Trimodal Fusion

Directory: `04_trimodal/`

Status: Folder created, model pending

To be added:

- `train_trimodal.py` - Combine all three modalities
- `trimodal_fusion.ipynb` - Training and evaluation
- `analysis.ipynb` - Explainability plots
- `config_trimodal.yaml` - Hyperparameters

### Inference and Deployment

Directory: `05_inference/`

Status: Folder created, scripts pending

To be added:

- `predict_sample.py` - Single sample prediction
- `predict_batch.py` - Batch predictions
- `streamlit_dashboard.py` - Interactive UI
- `xai_visualizations.py` - Explain predictions

### Data and Outputs

**data/** - Datasets

```
data/
├── Speech/
│   ├── read_text/ (HC, PD)
│   └── spontaneous/ (HC, PD)
├── Handwriting/
│   ├── healthy/
│   └── parkinson/
└── Gait/
    └── (all .txt files)
```

**outputs/** - Results and models

```
outputs/
├── gait_eda/
├── unimodal_speech/
├── unimodal_handwriting/
├── unimodal_gait/
├── bimodal_sg/
├── bimodal_hg/
├── bimodal_hs/
├── trimodal/
└── inference/
```

## Recommended Workflow

### Phase 1: Setup and EDA

1. Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. Set up conda environment
3. Install packages
4. Run gait EDA notebook: `jupyter lab 01_eda/gait_eda.ipynb`
5. Create speech and handwriting EDA notebooks

### Phase 2: Feature Extraction

1. Adapt unimodal training scripts from original project
2. Place in `02_unimodal/` with notebooks
3. Save outputs to `outputs/unimodal_*/`

### Phase 3: Bimodal Fusion

1. Adapt bimodal fusion scripts
2. Place in `03_bimodal/` with evaluation notebooks
3. Save outputs to `outputs/bimodal_*/`

### Phase 4: Trimodal and Analysis

1. Implement trimodal fusion
2. Generate explainability plots
3. Save final model and analysis

### Phase 5: Deployment

1. Create inference scripts
2. Build Streamlit dashboard
3. Document API usage

## Need Help

1. Setup issues: [QUICKSTART.md](QUICKSTART.md)
2. How to run notebooks: [01_eda/README.md](01_eda/README.md)
3. Understanding the project: [README.md](README.md)
4. Folder organization: [STRUCTURE.md](STRUCTURE.md)
5. Jupyter errors: check your conda environment is activated

Created: 2026-02-18
Status: Ready for Phase 1 (EDA)
