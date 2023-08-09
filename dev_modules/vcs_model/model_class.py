# Convolutional neural network model class.
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten

import sys
sys.path.insert(1, '../../')
from dev_modules.vcs_params import params_dataset


# MODEL HYPERPARAMETERS.
# convolution kernel size.
KERNEL_CONV_SIZE = 3    
# base number of filters.
N_FILTER_BASE = 2
# pooling kernel size.
KERNEL_POOL_SIZE = 2
# stride reduction factor
STRIDE_POOL = 2
# threshold decision.
THRESHOLD_DECISION = 0.5
# summary to logger.
TRAIN_HYPER = dict(kernel_conv_size = KERNEL_CONV_SIZE,    
                   n_filter_base = N_FILTER_BASE,
                   kernel_pool_size = KERNEL_POOL_SIZE,
                   stride_pool = STRIDE_POOL,
                   threshold_decision = THRESHOLD_DECISION)

class minimal_CNN_MCU():
    def __init__(self):
        """
        Construct net.
        """
        self.model = Sequential()
        # 128.
        self.model.add(Conv2D(filters=N_FILTER_BASE,
                              kernel_size=(KERNEL_CONV_SIZE, KERNEL_CONV_SIZE), strides=1,
                              activation='relu', padding='same',
                              input_shape=params_dataset.IMAGE_SIZE+ (1,)))
        self.model.add(AveragePooling2D(pool_size=(KERNEL_POOL_SIZE, KERNEL_POOL_SIZE),
                                        strides=STRIDE_POOL))
        # 64
        self.model.add(Conv2D(filters=2*N_FILTER_BASE,
                              kernel_size=(KERNEL_CONV_SIZE, KERNEL_CONV_SIZE), strides=1,
                              activation='relu', padding='same'))
        self.model.add(AveragePooling2D(pool_size=(KERNEL_POOL_SIZE, KERNEL_POOL_SIZE),
                                        strides=STRIDE_POOL))
        # 32
        self.model.add(Conv2D(filters=4*N_FILTER_BASE,
                              kernel_size=(KERNEL_CONV_SIZE, KERNEL_CONV_SIZE), strides=1,
                              activation='relu', padding='same'))
        self.model.add(AveragePooling2D(pool_size=(KERNEL_POOL_SIZE, KERNEL_POOL_SIZE),
                                        strides=STRIDE_POOL))
        self.model.add(Dropout(.05))
        # 16
        self.model.add(Conv2D(filters=4*N_FILTER_BASE,
                              kernel_size=(KERNEL_CONV_SIZE + 2, KERNEL_CONV_SIZE + 2), strides=1,
                              activation='relu', padding='same'))
        self.model.add(AveragePooling2D(pool_size=(KERNEL_POOL_SIZE, KERNEL_POOL_SIZE),
                                        strides=STRIDE_POOL))
        self.model.add(Dropout(.05))
        # 8
        self.model.add(Conv2D(filters=8*N_FILTER_BASE,
                              kernel_size=(KERNEL_CONV_SIZE, KERNEL_CONV_SIZE), strides=1,
                              activation='relu', padding='same'))
        self.model.add(Dropout(.05))
        # Output
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        
    def get_model(self) -> Sequential:
        return self.model
