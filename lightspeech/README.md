# LightSpeech: Lightweight Emotion Recognition for Resource-Constrained Environments

## Project Overview

LightSpeech addresses the challenge of deploying speech emotion recognition (SER) systems in resource-constrained environments. While existing SER systems achieve impressive accuracy using large transformer models like Wav2Vec2, they require substantial computational resources that make them impractical for mobile devices, embedded systems, or rural healthcare applications.

This project develops lightweight, efficient, and explainable emotion recognition models optimized for on-device deployment while maintaining competitive accuracy. Through advanced model compression techniques including knowledge distillation, quantization-aware training, and structured pruning, LightSpeech enables equitable AI access across diverse hardware environments.

## Key Innovations

- **Multi-Strategy Compression Pipeline**: Integrates knowledge distillation, quantization, and pruning in a unified framework
- **Hardware-Aware Optimization**: Benchmarks and optimizes for specific deployment targets (Raspberry Pi, Android, Jetson Nano)
- **Explainable Compression**: Provides interpretability analysis to understand how compressed models retain emotional features
- **Reproducible Research**: Complete workflow with versioned datasets, experiments, and configurations

## Project Goals

1. Develop an end-to-end lightweight SER pipeline that performs well under hardware constraints
2. Investigate model compression strategies including knowledge distillation, quantization-aware training, and structured pruning
3. Benchmark performance across multiple trade-offs: accuracy, latency, and model size
4. Provide interpretability analysis to explain how compressed models retain emotional features
5. Deliver a fully reproducible workflow portable to low-power devices

## Project Structure

```
lightspeech/
├── README.md              # This file
├── IMPLEMENTATION.md       # Detailed implementation guide
├── STRUCTURE.md            # Project structure explanation
├── requirements.txt        # Python dependencies
│
├── code/                  # Source code (organized by functionality)
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model architectures
│   ├── training/         # Training utilities
│   └── evaluation/       # Evaluation metrics
│
├── scripts/               # Main executable scripts (run in order)
│   ├── 01_prepare_data.py
│   ├── 02_train_baseline.py
│   ├── 03_compress_model.py
│   ├── 04_evaluate.py
│   └── 05_generate_plots.py
│
├── configs/               # Configuration files (YAML)
│   ├── baseline.yaml
│   ├── distillation.yaml
│   └── quantization.yaml
│
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets (download here)
│   ├── processed/        # Preprocessed audio
│   └── features/         # Extracted features
│
├── results/               # All outputs for technical report
│   ├── models/           # Trained model checkpoints
│   ├── plots/            # Figures and visualizations
│   ├── tables/           # Performance tables
│   └── logs/             # Training logs
│
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_interpretability.ipynb
│
└── report/                # Technical report materials
    ├── figures/          # Report figures
    ├── tables/           # Report tables
    └── draft/            # Report draft
```

See `STRUCTURE.md` for detailed explanation of the project structure.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

The project follows a simple step-by-step workflow. Run scripts in order:

```bash
# Step 1: Prepare data
python scripts/01_prepare_data.py --raw_data data/raw --output data/processed --augment  

# Step 2: Train baseline model
python scripts/02_train_baseline.py --data data/processed --output results/models --device cpu --epochs 30

# Step 3: Compress model
python scripts/03_compress_model.py --teacher_ckpt results/models/baseline_model.pth --output results/models --epochs 30 --distill --quantize --prune --data data/processed

# Step 4: Evaluate models
python scripts/04_evaluate.py --data data/processed --teacher_ckpt results/models/baseline_model.pth --student_ckpt results/models/models_distilled.pt --pruned_ckpt results/models/models_pruned.pt --quantized results/models/models_quantized_fallback.pt --output results --device cpu

# Step 5: Generate plots for report
python scripts/05_generate_plots.py
```

### Using Notebooks

For interactive analysis, use the Jupyter notebooks:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Results

All outputs are saved in the `results/` directory:
- `results/models/` - Trained model checkpoints
- `results/plots/` - Visualizations for technical report
- `results/tables/` - Performance metrics tables

## Datasets

- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song

## Research Contributions

This project contributes to the field through:

1. **Novel Compression Framework**: Unified approach combining multiple compression techniques
2. **Comprehensive Benchmarking**: Extensive evaluation across accuracy, latency, and model size
3. **Explainability Analysis**: Understanding of feature retention in compressed models
4. **Real-World Deployment**: Practical implementation on edge devices

## Expected Deliverables

- Trained baseline and compressed models
- Comparison tables (accuracy, size, latency)
- Interpretability analysis visuals with plots and figures
- Technical report (10-15 pages including results plots and analysis)
- Deployment demo (mobile/edge inference test)

## Contributing

This is a research project. For questions or contributions, please contact the project maintainer.

## License



## Acknowledgments

- CREMA-D, RAVDESS, and Emo-DB dataset creators
- HuggingFace Transformers community
- PyTorch and ONNX Runtime teams