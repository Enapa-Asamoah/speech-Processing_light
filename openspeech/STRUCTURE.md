# OpenSpeech Project Structure

This document explains the simplified project structure designed for easy navigation and implementation.

## Directory Overview

```
openspeech/
├── README.md                 # Project overview
├── IMPLEMENTATION.md         # Detailed implementation guide
├── STRUCTURE.md             # This file
├── requirements.txt          # Python dependencies
│
├── code/                     # All source code
│   ├── asr/                # Automatic Speech Recognition
│   ├── nlu/                # Natural Language Understanding
│   ├── tts/                # Text-to-Speech
│   └── integration/        # End-to-end integration
│
├── scripts/                  # Main executable scripts (run in order)
│   ├── 01_setup_asr.py      # Step 1: Setup ASR models
│   ├── 02_setup_nlu.py      # Step 2: Setup NLU system
│   ├── 03_setup_tts.py      # Step 3: Setup TTS models
│   ├── 04_integrate.py      # Step 4: Integrate components
│   ├── 05_evaluate.py       # Step 5: Evaluate pipeline
│   └── 06_demo.py           # Step 6: Run demo
│
├── configs/                  # Configuration files
│   ├── asr.yaml
│   ├── nlu.yaml
│   └── tts.yaml
│
├── data/                     # Dataset storage
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed data
│   └── transcriptions/      # ASR transcriptions
│
├── results/                   # All outputs
│   ├── models/              # Trained/fine-tuned models
│   ├── plots/               # Performance visualizations
│   ├── tables/              # Evaluation metrics
│   └── audio_samples/       # Generated TTS samples
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_asr_analysis.ipynb
│   ├── 02_nlu_analysis.ipynb
│   └── 03_tts_analysis.ipynb
│
└── report/                   # Technical report materials
    ├── figures/
    ├── tables/
    └── draft/
```

## Workflow

1. **Setup ASR** - Configure and fine-tune ASR models
2. **Setup NLU** - Configure intent classification and NER
3. **Setup TTS** - Configure text-to-speech synthesis
4. **Integration** - Integrate all components
5. **Evaluation** - Evaluate end-to-end performance
6. **Demo** - Create interactive demonstration

