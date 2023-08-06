# Training

Notebooks for training process:
1. Load dataset, augment, split in train and validation
2. Compile with hyperparameters and start training
3. Saved results in each run:
    - dataset division:
        - Train/Validation specific files
    - model hyperparameters:
        - subclassed model
        - pooling: kernel size, stride
        - convolution: kernel size, base number of filters
    - models: last epoch trained and best so far (smallest validation loss)
    - metrics:
        - loss
        - accuracy
        - precision
        - recall
        - false negatives
        - false positives
    - Example outputs:
        - images with wrong inference

# See results logged
## mlflow
Inside ```3_train/```:
```bash
$ mlflow ui
```

## tensorboard
Inside ```3_train/```:
```bash
$ tensorboard --logdir logs
```
