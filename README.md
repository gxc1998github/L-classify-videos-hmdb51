# classify-videos-hmdb51

Classify videos based on hmdb51 dataset in .avi format using CNN-LSTM python project tensorflow libraries

# Classify Videos HMDB51

This project aims to classify videos from the HMDB51 dataset using a CNN-LSTM model.

# Usage

python scripts/data_preprocessing.py
python scripts/train.py
python scripts/evaluate.py

## Project Structure

- `data/`
  - `raw/`: Contains the raw HMDB51 dataset.
  - `processed/`: Contains processed data ready for training.
- `notebooks/`: Jupyter notebooks for experimentation.
- `scripts/`
  - `data_preprocessing.py`: Script for data preprocessing.
  - `model.py`: Script defining the CNN-LSTM model.
  - `train.py`: Script to train the model.
  - `evaluate.py`: Script to evaluate the model.
- `config/`
  - `config.yaml`: Configuration file for hyperparameters and paths.
- `utils/`: Helper functions.
- `outputs/`
  - `models/`: Saved model checkpoints.
  - `logs/`: Training and evaluation logs.
  - `results/`: Evaluation results and metrics.
- `tests/`: Unit tests for scripts and functions.
