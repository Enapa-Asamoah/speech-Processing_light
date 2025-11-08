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
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures (baseline, compressed)
│   ├── training/          # Training scripts and loops
│   ├── compression/       # Compression techniques (distillation, quantization, pruning)
│   ├── evaluation/        # Evaluation metrics and benchmarking
│   ├── deployment/        # Deployment utilities and conversion
│   └── utils/             # Utility functions
├── configs/               # Configuration files (YAML)
├── experiments/           # Experiment tracking and results
│   ├── baseline/         # Baseline model experiments
│   ├── compressed/       # Compressed model experiments
│   ├── distilled/        # Knowledge distillation experiments
│   └── quantized/        # Quantization experiments
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed audio
│   └── features/         # Extracted features
├── notebooks/             # Jupyter notebooks
│   ├── exploration/      # Data exploration
│   ├── analysis/         # Results analysis
│   └── interpretability/ # Model interpretability
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
│   ├── paper/            # Research paper drafts
│   ├── api/              # API documentation
│   └── deployment/       # Deployment guides
├── scripts/               # Standalone scripts
├── outputs/               # Model outputs, logs, plots
└── deployment/           # Deployment configurations
    ├── raspberry_pi/
    ├── android/
    └── jetson/
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download datasets (CREMA-D, RAVDESS, or Emo-DB)
2. Run preprocessing pipeline:
```bash
python scripts/preprocessing/prepare_data.py --dataset CREMA-D --output_dir data/processed
```

### Training Baseline Model

```bash
python scripts/training/train_baseline.py --config configs/baseline_config.yaml
```

### Model Compression

```bash
# Knowledge Distillation
python scripts/training/train_distilled.py --config configs/distillation_config.yaml

# Quantization
python scripts/training/train_quantized.py --config configs/quantization_config.yaml

# Pruning
python scripts/training/train_pruned.py --config configs/pruning_config.yaml
```

## Datasets

- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **Emo-DB**: Berlin Database of Emotional Speech

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

[Specify license]

## Acknowledgments

- CREMA-D, RAVDESS, and Emo-DB dataset creators
- HuggingFace Transformers community
- PyTorch and ONNX Runtime teams