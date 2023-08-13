# Convolutional neural network model class.
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.insert(1, '../../')
from dev_modules.vcs_params import params_dataset
from dev_modules.vcs_params import params_model

import cv2
import numpy as np
from src.Lite_handle import Lite_handler

class fp_CNN_MCU():
    """
    Floating point CNN for IoT and MCU devices.
    """
    def __init__(self):
        """
        Construct net.
        """
        self.model_classes = ['Usable', 'Defective']
        self.threshold_decision = params_model.THRESHOLD_DECISION
        
        self.fp_model = Sequential()
        # 128.
        self.fp_model.add(Conv2D(
            filters=params_model.L0_N_FILTER,
            kernel_size=(params_model.L0_KERNEL_SIZE,
                         params_model.L0_KERNEL_SIZE),
            strides=1,
            activation='relu', padding='same',
            input_shape=params_dataset.IMAGE_SIZE + (1,)))
        self.fp_model.add(MaxPooling2D(
            pool_size=(params_model.POOL_KERNEL_SIZE,
                       params_model.POOL_KERNEL_SIZE),
            strides=params_model.POOL_STRIDE))
        # 64
        self.fp_model.add(Conv2D(
            filters=params_model.L1_N_FILTER,
            kernel_size=(params_model.L1_KERNEL_SIZE,
                         params_model.L1_KERNEL_SIZE),
            strides=1,
            activation='relu', padding='same'))
        self.fp_model.add(MaxPooling2D(
            pool_size=(params_model.POOL_KERNEL_SIZE,
                       params_model.POOL_KERNEL_SIZE),
            strides=params_model.POOL_STRIDE))
        # 32
        self.fp_model.add(Conv2D(
            filters=params_model.L2_N_FILTER,
            kernel_size=(params_model.L2_KERNEL_SIZE,
                         params_model.L2_KERNEL_SIZE),
            strides=1,
            activation='relu', padding='same'))
        self.fp_model.add(MaxPooling2D(
            pool_size=(params_model.POOL_KERNEL_SIZE,
                       params_model.POOL_KERNEL_SIZE),
            strides=params_model.POOL_STRIDE))
        self.fp_model.add(Dropout(.05))
        # 16
        self.fp_model.add(Conv2D(
            filters=params_model.L3_N_FILTER,
            kernel_size=(params_model.L3_KERNEL_SIZE,
                         params_model.L3_KERNEL_SIZE),
            strides=1,
            activation='relu', padding='same'))
        self.fp_model.add(MaxPooling2D(
            pool_size=(params_model.POOL_KERNEL_SIZE,
                       params_model.POOL_KERNEL_SIZE),
            strides=params_model.POOL_STRIDE))
        self.fp_model.add(Dropout(.05))
        # 8
        self.fp_model.add(Conv2D(
            filters=params_model.L4_N_FILTER,
            kernel_size=(params_model.L4_KERNEL_SIZE,
                         params_model.L4_KERNEL_SIZE),
            strides=1,
            activation='relu', padding='same'))
        self.fp_model.add(Dropout(.05))
        # Output
        self.fp_model.add(Flatten())
        self.fp_model.add(Dense(1, activation='sigmoid'))


    def get_model(self) -> Sequential:
        """
        Return the fp model structure.
        """
        return self.fp_model


    def load_model(self, path: str) -> tf.keras.models.Sequential:
        """
        Load trained model.
        """
        self.loaded_model = tf.keras.models.load_model(path)
        return self.loaded_model

    def fp_predict(self, x_input: object) -> tf.Tensor:
        """
        Abstraction to inference in floating point model.
        """
        return self.loaded_model.predict(x_input)


class qt_CNN_MCU():
    """
    Quantized CNN for IoT and MCU devices.
    """
    def __init__(self):
        """
        Init lite handler class.
        """
        self.lite_h = Lite_handler()


    def get_model(self,
                  fp_model: Sequential,
                  dataset: ImageDataGenerator,
                  savedir: str) -> Sequential:
        """
        Return the qt model build.
        """
        self.qt_model = self.lite_h.build_quantized_model(fp_model, dataset, savedir)
        return self.qt_model


    def qt_predict(self, dataset: ImageDataGenerator) -> tf.Tensor:
        """
        Abstraction to inference in quantized model.
        """
        y_pred = list()
        for file in dataset.filepaths:
            sample = cv2.imread(file, cv2.IMREAD_GRAYSCALE).reshape(-1, 128, 128, 1)
            
            y_pred.append(self.lite_h.predict_tflite(sample)[0])
        y_pred = np.array(y_pred)
        
        return y_pred