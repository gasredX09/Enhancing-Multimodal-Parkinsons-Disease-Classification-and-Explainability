# BiomedAI - Multimodal Parkinson Prediction

This repository contains:

- The original reference project in `Multimodal-Parkinson-Disease-Prediction-With-XAI/`.
- Our structured replication work in `project/replica/`.

## Quick Start (Conda)

```bash
conda create -n biomedai python=3.10
conda activate biomedai
pip install -r requirements.txt
```

Open the gait EDA notebook:

```bash
cd project/replica/01_eda
jupyter lab gait_eda.ipynb
```

## Where to Look

- Replication docs: `project/replica/README.md`
- Setup guide: `project/replica/QUICKSTART.md`
- Gait EDA notebook: `project/replica/01_eda/gait_eda.ipynb`
- Results and outputs: `project/replica/outputs/`

## Data

Datasets live in `project/replica/data/`. The current pipeline uses the IMU gait files. Speech and handwriting datasets are added later for multimodal fusion.

## Contributing

See `CONTRIBUTING.md` for workflow and conventions.
