# Speech2Health: Detecting Stress and Fatigue from Speech Prosody

## Project Overview

Speech2Health bridges the gap between speech processing and healthcare applications. Human speech carries rich physiological and emotional cues—including pitch, energy, and vocal stability—that change under stress or fatigue. While most speech emotion recognition models focus on categorical emotions (e.g., anger, joy), Speech2Health focuses on the detection of psychophysiological states such as stress and fatigue from acoustic-prosodic speech features.

This project develops interpretable machine learning models that can detect stress and fatigue non-invasively, enabling applications in mental health monitoring, occupational safety, and well-being assessment. By ensuring transparency, reproducibility, and real-world relevance, Speech2Health contributes to the growing field of audio-based health research.

## Key Innovations

- **Psychophysiological State Detection**: Focus on stress and fatigue rather than categorical emotions
- **Comprehensive Feature Engineering**: Extraction of acoustic-prosodic biomarkers (pitch, jitter, shimmer, formants)
- **Interpretable Models**: Classical ML and neural approaches with explainability
- **Biomarker Identification**: Key vocal indicators of stress and fatigue
- **Clinical Relevance**: Models designed for healthcare and monitoring applications

## Project Goals

1. Develop interpretable models for detecting stress and fatigue from acoustic features
2. Compare classical ML (Random Forest, XGBoost) and neural approaches (CNN-LSTM)
3. Identify key vocal biomarkers associated with stress and fatigue
4. Evaluate model generalization across speakers and sessions
5. Promote reproducibility and explainability in audio-based health research

## Project Structure

```
speech2health/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── feature_extraction/ # Acoustic-prosodic feature extraction
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation metrics
│   ├── interpretability/  # Model interpretability
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── experiments/           # Experiment tracking
│   ├── classical_ml/     # Classical ML experiments
│   ├── neural_networks/   # Neural network experiments
│   └── ensemble/         # Ensemble model experiments
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed audio
│   ├── features/         # Extracted features
│   └── biomarkers/       # Identified biomarkers
├── notebooks/             # Jupyter notebooks
│   ├── exploration/      # Data exploration
│   ├── feature_analysis/ # Feature importance analysis
│   └── model_comparison/ # Model comparison
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
│   ├── paper/            # Research paper drafts
│   ├── clinical_validation/ # Clinical validation reports
│   └── biomarker_analysis/ # Biomarker documentation
├── scripts/               # Standalone scripts
├── outputs/               # Model outputs, logs, plots
└── demos/                 # Interactive demos
    ├── flask/            # Flask web app
    └── gradio/           # Gradio interface
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

1. Download datasets (DAIC-WOZ, AVEC 2019, or SEMAINE)
2. Run preprocessing pipeline:
```bash
python scripts/preprocessing/prepare_data.py --dataset DAIC-WOZ --output_dir data/processed
```

### Feature Extraction

```bash
python scripts/feature_extraction/extract_features.py --input_dir data/processed --output_dir data/features
```

### Training Models

```bash
# Classical ML (Random Forest)
python scripts/training/train_classical.py --model random_forest --config configs/random_forest_config.yaml

# Neural Network (CNN-LSTM)
python scripts/training/train_neural.py --model cnn_lstm --config configs/cnn_lstm_config.yaml

# XGBoost
python scripts/training/train_classical.py --model xgboost --config configs/xgboost_config.yaml
```

### Evaluation

```bash
python scripts/evaluation/evaluate_models.py --model_dir outputs/models --test_data data/features/test
```

## Datasets

- **DAIC-WOZ**: Clinical interviews for depression/stress detection
- **AVEC 2019 Challenge Dataset**: Multimodal affect recognition
- **SEMAINE Database**: Emotional speech with stress annotations

## Research Contributions

This project contributes to the field through:

1. **Biomarker Discovery**: Identification of vocal indicators of stress and fatigue
2. **Model Comparison**: Comprehensive evaluation of classical ML vs neural approaches
3. **Interpretability**: Explainable models suitable for clinical contexts
4. **Generalization Analysis**: Cross-speaker and cross-session evaluation
5. **Reproducible Research**: Complete workflow for audio-based health research

## Expected Deliverables

- Trained stress/fatigue detection models
- Feature importance and interpretability analyses
- Performance and statistical comparison tables with visualizations
- Technical report (10-15 pages including results plots, biomarker analysis, and model comparisons)
- Reproducibility documentation (scripts, configs, seeds)
- Optional interactive demo (Flask/Gradio)

## Clinical Applications

- **Mental Health Monitoring**: Early detection of stress and fatigue
- **Occupational Safety**: Workplace stress assessment
- **Well-being Assessment**: Personal health tracking
- **Telemedicine**: Remote health monitoring

## Contributing

This is a research project with potential healthcare applications. Contributions that improve interpretability, clinical relevance, or reproducibility are welcome.

## License

[Specify license]

## Acknowledgments

- DAIC-WOZ, AVEC, and SEMAINE dataset creators
- Clinical research collaborators
- Open-source audio processing community

