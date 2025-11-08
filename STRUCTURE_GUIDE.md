# Project Structure Guide

This document explains the simplified, student-friendly structure used across all four projects.

## Design Principles

The project structures have been redesigned with the following principles:

1. **Simplicity**: Clear, linear workflow that's easy to follow
2. **Clarity**: Numbered scripts show execution order
3. **Organization**: Related code grouped together
4. **Publication-Ready**: All results in one place for easy report generation
5. **Student-Friendly**: Less complexity, more guidance

## Common Structure Pattern

All projects follow this simplified structure:

```
project_name/
├── README.md              # Project overview
├── IMPLEMENTATION.md      # Detailed guide
├── STRUCTURE.md          # Structure explanation
├── requirements.txt       # Dependencies
│
├── code/                  # Source code (organized by functionality)
├── scripts/               # Numbered scripts (run in order: 01, 02, 03...)
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── results/               # All outputs (models, plots, tables)
├── notebooks/             # Jupyter notebooks for analysis
└── report/                # Technical report materials
```

## Key Improvements

### 1. Simplified Code Organization
- **Before**: Complex nested `src/` with many subdirectories
- **After**: Clean `code/` directory organized by functionality
- **Benefit**: Easier to find and understand code

### 2. Numbered Scripts
- **Before**: Scripts scattered in subdirectories
- **After**: Numbered scripts in `scripts/` (01, 02, 03...)
- **Benefit**: Clear execution order, no confusion

### 3. Unified Results Directory
- **Before**: Outputs scattered (outputs/, experiments/, etc.)
- **After**: Single `results/` directory with subdirectories
- **Benefit**: Everything for technical report in one place

### 4. Clear Documentation
- **Before**: Documentation mixed with code
- **After**: Separate `STRUCTURE.md` explains organization
- **Benefit**: Students understand structure quickly

### 5. Report-Focused
- **Before**: Production deployment focus
- **After**: Technical report focus with `report/` directory
- **Benefit**: Clear path to publication-ready output

## Workflow Example

For any project, students follow this simple workflow:

1. **Setup**: Install dependencies
2. **Data**: Run `01_prepare_data.py`
3. **Train**: Run `02_train_baseline.py`
4. **Process**: Run subsequent numbered scripts
5. **Analyze**: Use notebooks for exploration
6. **Report**: Gather results from `results/` directory

## Benefits for Students

1. **Less Overwhelming**: Simpler structure reduces cognitive load
2. **Clear Path**: Numbered scripts show exactly what to do next
3. **Easy Navigation**: Logical organization makes finding files easy
4. **Report Ready**: All outputs organized for technical report
5. **Template Files**: Starter code shows what to implement

## Migration Notes

If you have existing code in the old structure:
- Move source code from `src/` to `code/`
- Consolidate scripts into `scripts/` with numbering
- Move outputs to `results/` subdirectories
- Update imports to reflect new structure

## Project-Specific Variations

While all projects follow the same pattern, each has slight variations:

- **LightSpeech**: Focus on compression techniques
- **FairVoice**: Additional bias analysis and reporting
- **Speech2Health**: Feature extraction emphasis
- **OpenSpeech**: Component-based (ASR, NLU, TTS)

See individual `STRUCTURE.md` files in each project for details.

