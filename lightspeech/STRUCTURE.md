# LightSpeech Project Structure

This document explains the simplified project structure designed for easy navigation and implementation.

## Directory Overview

```
lightspeech/
├── README.md                 # Project overview and quick start
├── IMPLEMENTATION.md         # Detailed implementation guide
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── code/                     # All source code
│   ├── __init__.py
│   ├── data/               # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py       # Dataset loading
│   │   ├── preprocess.py   # Audio preprocessing
│   │   └── augment.py      # Data augmentation
│   ├── models/             # Model architectures
│   │   ├── __init__.py
│   │   ├── baseline.py    # Baseline CNN/Transformer
│   │   ├── student.py      # Lightweight student model
│   │   └── compression.py  # Compression utilities
│   ├── training/           # Training scripts
│   │   ├── __init__.py
│   │   ├── trainer.py      # Base trainer
│   │   ├── distillation.py # Knowledge distillation
│   │   └── quantization.py # Quantization training
│   └── evaluation/         # Evaluation metrics
│       ├── __init__.py
│       ├── metrics.py      # Accuracy, F1, etc.
│       ├── efficiency.py   # Latency, model size
│       └── explainability.py # SHAP, Grad-CAM
│
├── scripts/                  # Main executable scripts
│   ├── 01_prepare_data.py   # Step 1: Prepare dataset
│   ├── 02_train_baseline.py # Step 2: Train baseline
│   ├── 03_compress_model.py # Step 3: Compress model
│   ├── 04_evaluate.py       # Step 4: Evaluate models
│   └── 05_generate_plots.py # Step 5: Generate visualizations
│
├── configs/                  # Configuration files
│   ├── baseline.yaml        # Baseline model config
│   ├── distillation.yaml    # Distillation config
│   └── quantization.yaml    # Quantization config
│
├── data/                     # Dataset storage
│   ├── raw/                 # Original datasets (not in git)
│   ├── processed/           # Preprocessed audio
│   └── features/            # Extracted features
│
├── results/                  # All outputs for technical report
│   ├── models/              # Trained model checkpoints
│   ├── plots/               # Figures and visualizations
│   ├── tables/               # Performance tables
│   └── logs/                # Training logs
│
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_interpretability.ipynb
│
└── report/                   # Technical report materials
    ├── figures/              # Report figures
    ├── tables/               # Report tables
    └── draft/                # Report draft (LaTeX/Markdown)
```

## Workflow

The project follows a simple linear workflow:

1. **Data Preparation** (`scripts/01_prepare_data.py`)
   - Downloads and preprocesses datasets
   - Extracts features
   - Creates train/val/test splits

2. **Baseline Training** (`scripts/02_train_baseline.py`)
   - Trains full-size baseline model
   - Saves checkpoint to `results/models/`

3. **Model Compression** (`scripts/03_compress_model.py`)
   - Applies compression techniques
   - Saves compressed models

4. **Evaluation** (`scripts/04_evaluate.py`)
   - Evaluates all models
   - Generates performance metrics
   - Saves results to `results/tables/`

5. **Visualization** (`scripts/05_generate_plots.py`)
   - Creates plots for technical report
   - Saves to `results/plots/`

## Key Files

- **Entry Point Scripts**: All in `scripts/` directory, numbered for order
- **Source Code**: Organized by functionality in `code/`
- **Results**: Everything goes to `results/` for easy report generation
- **Notebooks**: For interactive analysis and exploration

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run scripts in order: `python scripts/01_prepare_data.py`
3. Check results in `results/` directory
4. Use notebooks for detailed analysis

