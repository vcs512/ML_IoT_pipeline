# Convolutional neural network model class.
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten

import sys
sys.path.insert(1, '../../')
from dev_modules.vcs_params import params_dataset
from dev_modules.vcs_params import params_model


class fp_CNN_MCU():
    def __init__(self):
        """
        Construct net.
        """
        self.model_classes = ['Usable', 'Defective']
        self.threshold_decision = params_model.THRESHOLD_DECISION
        
        self.fp_model = Sequential()
        # 128.
        self.fp_model.add(Conv2D(filters=params_model.N_FILTER_BASE,
                              kernel_size=(params_model.KERNEL_CONV_SIZE,
                                           params_model.KERNEL_CONV_SIZE),
                              strides=1,
                              activation='relu', padding='same',
                              input_shape=params_dataset.IMAGE_SIZE+ (1,)))
        self.fp_model.add(AveragePooling2D(pool_size=(params_model.KERNEL_POOL_SIZE,
                                                   params_model.KERNEL_POOL_SIZE),
                                        strides=params_model.STRIDE_POOL))
        # 64
        self.fp_model.add(Conv2D(filters=2*params_model.N_FILTER_BASE,
                              kernel_size=(params_model.KERNEL_CONV_SIZE,
                                           params_model.KERNEL_CONV_SIZE),
                              strides=1,
                              activation='relu', padding='same'))
        self.fp_model.add(AveragePooling2D(pool_size=(params_model.KERNEL_POOL_SIZE,
                                                   params_model.KERNEL_POOL_SIZE),
                                        strides=params_model.STRIDE_POOL))
        # 32
        self.fp_model.add(Conv2D(filters=4*params_model.N_FILTER_BASE,
                              kernel_size=(params_model.KERNEL_CONV_SIZE,
                                           params_model.KERNEL_CONV_SIZE),
                              strides=1,
                              activation='relu', padding='same'))
        self.fp_model.add(AveragePooling2D(pool_size=(params_model.KERNEL_POOL_SIZE,
                                                   params_model.KERNEL_POOL_SIZE),
                                        strides=params_model.STRIDE_POOL))
        self.fp_model.add(Dropout(.05))
        # 16
        self.fp_model.add(Conv2D(filters=4*params_model.N_FILTER_BASE,
                              kernel_size=(params_model.KERNEL_CONV_SIZE,
                                           params_model.KERNEL_CONV_SIZE),
                              strides=1,
                              activation='relu', padding='same'))
        self.fp_model.add(AveragePooling2D(pool_size=(params_model.KERNEL_POOL_SIZE,
                                                   params_model.KERNEL_POOL_SIZE),
                                        strides=params_model.STRIDE_POOL))
        self.fp_model.add(Dropout(.05))
        # 8
        self.fp_model.add(Conv2D(filters=8*params_model.N_FILTER_BASE,
                              kernel_size=(params_model.KERNEL_CONV_SIZE,
                                           params_model.KERNEL_CONV_SIZE),
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