# FairVoice Project Structure

This document explains the simplified project structure designed for easy navigation and implementation.

## Directory Overview

```
fairvoice/
├── README.md                 # Project overview and quick start
├── IMPLEMENTATION.md         # Detailed implementation guide
├── STRUCTURE.md             # This file
├── requirements.txt          # Python dependencies
│
├── code/                     # All source code
│   ├── data/               # Data loading with demographic metadata
│   ├── models/             # Model architectures
│   ├── training/           # Training scripts
│   ├── bias_mitigation/    # Bias mitigation techniques
│   ├── explainability/     # Explainability tools (SHAP, Grad-CAM)
│   └── evaluation/         # Evaluation and fairness metrics
│
├── scripts/                  # Main executable scripts (run in order)
│   ├── 01_prepare_data.py   # Step 1: Prepare data with metadata
│   ├── 02_train_baseline.py  # Step 2: Train baseline model
│   ├── 03_assess_bias.py     # Step 3: Assess bias in baseline
│   ├── 04_mitigate_bias.py   # Step 4: Apply bias mitigation
│   ├── 05_evaluate_fairness.py # Step 5: Evaluate fairness
│   └── 06_generate_report.py  # Step 6: Generate bias report
│
├── configs/                  # Configuration files
│   ├── baseline.yaml
│   ├── fairness.yaml
│   └── adversarial.yaml
│
├── data/                     # Dataset storage
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed audio
│   └── metadata/            # Demographic metadata
│
├── results/                 # All outputs
│   ├── models/              # Trained models
│   ├── plots/               # Visualizations
│   ├── tables/              # Performance and bias metrics
│   └── reports/             # Bias assessment reports
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_bias_analysis.ipynb
│   ├── 02_explainability.ipynb
│   └── 03_fairness_comparison.ipynb
│
└── report/                   # Technical report materials
    ├── figures/
    ├── tables/
    └── draft/
```

## Workflow

1. **Data Preparation** - Load data and extract demographic metadata
2. **Baseline Training** - Train standard SER model
3. **Bias Assessment** - Measure bias across demographics
4. **Bias Mitigation** - Apply mitigation strategies
5. **Fairness Evaluation** - Evaluate fairness metrics
6. **Report Generation** - Create comprehensive bias report

