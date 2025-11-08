# Scripts Directory

This directory contains the main executable scripts for the LightSpeech project. Run them in numerical order.

## Script Execution Order

1. **01_prepare_data.py** - Prepare and preprocess dataset
2. **02_train_baseline.py** - Train baseline emotion recognition model
3. **03_compress_model.py** - Apply compression techniques
4. **04_evaluate.py** - Evaluate all models
5. **05_generate_plots.py** - Generate visualizations for report

## Usage

Each script can be run from the project root:

```bash
python scripts/01_prepare_data.py --dataset CREMA-D
```

Use `--help` flag to see available options for each script.

## Implementation Notes

These scripts are starter templates. Students should:
- Implement the TODO sections
- Add error handling
- Add logging
- Customize based on their specific dataset and requirements

