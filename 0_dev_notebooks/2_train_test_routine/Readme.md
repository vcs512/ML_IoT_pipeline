# Train-test routine


# Training
Training process:
1. Load dataset, augment, split in train and validation
1. Compile with hyperparameters and start training
1. Saved results in each run:
    - Dataset division:
        - Train/Validation specific files
    - Model hyperparameters:
        - Model
        - Pooling: kernel size, stride
        - Convolution: kernel size, base number of filters
    - Models: last epoch trained and best so far (smallest validation loss)
    - Metrics:
        - Loss
        - Accuracy
        - Precision
        - Recall
    - Example outputs:
        - Images with wrong inference


# MCU/IoT model
Convert full floating point model to run in IoT/MCU:
- Quantize: check errors and effects


# Test
Load models and evaluate through test set for:
- Accuracy
- Precision
- Recall
- False negatives

# See results logged
## mlflow
```bash
$ mlflow ui
```
## tensorboard
```bash
$ tensorboard --logdir logs
```