# ML for IoT pipeline - vcs512

Pipeline to train and evaluate metrics of Convolutional Neural Networks (CNNs) 
that fit in IoT/MCU (Internet of Things / Microcontroller) devices.

## ML Pipeline

Directories were made to follow a standard Machine Learning (ML) development to 
production pipeline:
1. [DATASET](./0_dev_notebooks/1_dataset/Readme.md):
    Save the dataset used for training and test
2. [EDA](./0_dev_notebooks/2_EDA_Exploratory_Data_Analysis/Readme.md):
    Perform statistical and qualitative analysis in the dataset
3. [TRAIN](./0_dev_notebooks/3_train/Readme.md):
    Train a model, tune hyperparameters, cross-validate
4. [LITE_MODEL](./0_dev_notebooks/4_lite_model/Readme.md):
    Quantize a model and get a binary to embed in IoT/MCU dispositives
5. [TEST](./0_dev_notebooks/5_test/Readme.md):
    Confirm training trends, look for over/underfitting

## Requirements install
```bash
$ pip install -r requirements.txt
```

# References
## Dataset reference
- https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

## General pipeline
- https://www.kaggle.com/code/ginsaputra/visual-inspection-of-casting-products-using-cnn/notebook
- https://www.kaggle.com/code/sfbruno/image-processing-cnn

## TinyML
- https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb
- WARDEN, Pete; SITUNAYAKE, Daniel. **TinyML**. O'Reilly Media, Incorporated, 2019.
