# Development notebooks
Used for interactively test modules created.


# Exploratory data analysis (EDA)
[EDA_Notebook](./2_dataset_explore.ipynb)

Notebooks to perform EDA:
- analyse dataset
- view trends and stratification in data
- verify pre-processing necessities
- define metrics that will be of interest

## Observations
### Complete set
| **Class**     | **Samples** | **total (%)** |
|---------------|:-----------:|:-----------:|
| 0 (usable)    |     517     |    39.86    |
| 1 (defective) |     780     |    60.14    |
| **TOTAL**     |   **1297**  |  **100.00** |

### Sets division
| **Set**   | **Samples** | **total (%)** |
|-----------|:-----------:|:-----------:|
| Train     |     1114    |    85.89    |
| Test      |     183     |    14.11    |
| **TOTAL** |   **1297**  |  **100.00** |

### Data
Unbalance between usable (40%) and defective (60%).
Important metrics to check unbalance efects:
- Recall
- Precision
- Confusion metrix

Useful data augmentation:
- Vertical/horizontal flip.

Data already has different zoom, background colors and angulation.


# Training
[Training_notebook](./3_holdout_train.ipynb)

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

## See results logged
### mlflow
```bash
$ mlflow ui
```
### tensorboard
```bash
$ tensorboard --logdir logs
```


# MCU/IoT model
[Lite_model_notebook](./4_lite_model/create_lite_model.ipynb)

Convert full floating point model to run in IoT/MCU:
- Quantize: check errors and efects


# Test
[Tests_notebook](./5_test_results.ipynb)

Load models and evaluate through test set for:
- Accuracy
- Precision
- Recall
- False negatives
