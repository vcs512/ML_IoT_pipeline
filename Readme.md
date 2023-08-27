# ML for IoT pipeline - vcs512

Pipeline to train and evaluate metrics of Convolutional Neural Networks (CNNs) 
that fit in IoT/MCU (Internet of Things / Microcontroller) devices.

## ML Pipeline

Notebooks were made to follow a standard Machine Learning (ML)
development to production pipeline:
- [COMPLETE_INFORMATION](./0_dev_notebooks/Readme.md):
    Complete general information concluded on each step
1. [DATASET](./dataset/Readme.md):
    Save the dataset used for training and test
2. [EDA](./0_dev_notebooks/2_dataset_explore.ipynb):
    Perform statistical and qualitative analysis in the dataset
3. [TRAIN](./0_dev_notebooks/3_holdout_train.ipynb):
    Train a model, tune hyperparameters, evaluate results
4. [LITE_MODEL](./0_dev_notebooks/4_fp_qt_train.ipynb):
    Quantize a model and get a binary to embed in IoT/MCU dispositives
5. [TEST](./0_dev_notebooks/5_fp_qt_train_test.ipynb):
    Confirm training trends, look for over/underfitting
6. [MICRO_MODEL](./0_dev_notebooks/9_convert_to_TFLiteMicro.ipynb):
    Convert the TensorFlow Lite model to a TF Lite Micro

## Automatic scripts
For automatic usage in CLI, it is possible to run scripts in ```run``` directory.

## Requirements installation
```bash
$ pip install -r requirements.txt
```

# References

# Implementarion of a model in an ESP32 MCU:
- https://github.com/vcs512/micro-cnn

## Dataset reference
- https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

## General pipeline
- https://www.kaggle.com/code/ginsaputra/visual-inspection-of-casting-products-using-cnn/notebook
- https://www.kaggle.com/code/sfbruno/image-processing-cnn

## TinyML
- https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb
- WARDEN, Pete; SITUNAYAKE, Daniel. **TinyML**. O'Reilly Media, Incorporated, 2019.
