# Speech2Health Project Structure

This document explains the simplified project structure designed for easy navigation and implementation.

## Directory Overview

```
speech2health/
├── README.md                 # Project overview
├── IMPLEMENTATION.md         # Detailed implementation guide
├── STRUCTURE.md             # This file
├── requirements.txt          # Python dependencies
│
├── code/                     # All source code
│   ├── data/               # Data loading and preprocessing
│   ├── feature_extraction/  # Acoustic-prosodic features
│   ├── models/             # Model architectures (ML and neural)
│   ├── training/           # Training utilities
│   ├── evaluation/         # Evaluation metrics
│   └── interpretability/   # Biomarker analysis
│
├── scripts/                  # Main executable scripts (run in order)
│   ├── 01_prepare_data.py   # Step 1: Prepare audio data
│   ├── 02_extract_features.py # Step 2: Extract features
│   ├── 03_train_classical.py  # Step 3: Train classical ML
│   ├── 04_train_neural.py     # Step 4: Train neural networks
│   ├── 05_evaluate.py        # Step 5: Evaluate all models
│   └── 06_identify_biomarkers.py # Step 6: Identify biomarkers
│
├── configs/                  # Configuration files
│   ├── random_forest.yaml
│   ├── xgboost.yaml
│   └── cnn_lstm.yaml
│
├── data/                     # Dataset storage
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed audio
│   └── features/            # Extracted features
│
├── results/                   # All outputs
│   ├── models/              # Trained models
│   ├── plots/               # Visualizations
│   ├── tables/              # Performance tables
│   └── biomarkers/          # Biomarker analysis results
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_feature_analysis.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_biomarker_visualization.ipynb
│
└── report/                   # Technical report materials
    ├── figures/
    ├── tables/
    └── draft/
```

## Workflow

1. **Data Preparation** - Load and preprocess audio
2. **Feature Extraction** - Extract acoustic-prosodic features
3. **Train Classical ML** - Train Random Forest, XGBoost, etc.
4. **Train Neural Networks** - Train CNN-LSTM models
5. **Evaluation** - Compare all models
6. **Biomarker Identification** - Identify key vocal indicators

