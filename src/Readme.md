# Abstration classes

## [Trainer](./Trainer.py)
Operate with datasets:
- Define sets for training, validation and test
- Call FP (train) and QT (build) model operations
- Call metrics handler

## [Logger](./Logger.py)
Ensure logging results and reproducibility:
- Handler to save *mlflow* objects produced

## [Metrics](./Metrics.py)
Calculate mathematical metrics:
- Confusion matrix
- Quantization errors

## [Lite_handle](./Lite_handle.py)
Abstract *TensorFlow Lite* operations:
- Build QT model from representative set
- Predict routine for QT model