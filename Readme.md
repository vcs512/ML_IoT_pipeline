# ML for IoT pipeline - vcs512

Pipeline to train and evaluate metrics of Convolutional Neural Networks (CNNs) 
that fit in IoT/MCU (Internet of Things / Microcontroller) devices.

## ML Pipeline
Notebooks were made to follow a standard Machine Learning (ML)
development to production pipeline:
- [COMPLETE_INFORMATION](./0_dev_notebooks/):
    Complete general information concluded on each step

## Automatic scripts
For automatic usage in CLI, it is possible to run scripts in [scripts](./src/scripts/fp_qt_train_test_routine.py).


## Model and general parameters
Module that gather model and general parameters used: [development module](./dev_modules/)

## Model and metrics results
Directory to save the models and its results: [model dir](./model/)

## Requirements installation
```bash
$ pip install -r requirements.txt
```

# References

## Implementation of a model in an ESP32 MCU:
- https://github.com/vcs512/micro-cnn

## Dataset reference
- https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

## General pipeline
- https://www.kaggle.com/code/ginsaputra/visual-inspection-of-casting-products-using-cnn/notebook
- https://www.kaggle.com/code/sfbruno/image-processing-cnn

## TinyML
- https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb
- WARDEN, Pete; SITUNAYAKE, Daniel. **TinyML**. O'Reilly Media, Incorporated, 2019.
