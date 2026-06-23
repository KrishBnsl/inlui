# Reproducibility Guide

The codebase is built to ensure maximum reproducibility across execution environments.

## Environment Details
- **Python Version**: 3.12.x
- **OS Assumption**: Linux/macOS preferred (Windows via WSL)
- **Random Seed**: `42` (Fixed for Monte Carlo, Permutation Importance, and splits)

## Environment Setup
A strict environment file is provided for Conda/Mamba users.

```bash
conda env create -f environment.yml
conda activate josaa_analytics
```
Alternatively, using `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r models/requirements.txt
```

## Dataset Assumptions
- The `data/raw/` folder must contain historical CSV files structured as `YYYY_round_R.csv`.
- Features expect uniform naming conventions pre-2022 and post-2022 (addressed in the Data Layer).

## Pipeline Execution
Run the layers sequentially to build the artifacts from scratch:
1. `python models/preprocessing.py`
2. `python models/eda_features.py`
3. `python models/train_models.py`
4. `python models/recommendation.py`
5. `python models/evaluation.py`
