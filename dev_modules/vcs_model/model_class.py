# Convolutional neural network model class.
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten


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


class minimal_CNN_MCU(tf.keras.Model):
    """
    Micro-CNN model for use in ESP32.
    """
    
    def __init__(self, img_size: tuple) -> None:
        """
        Model architecture.
        """
        super().__init__(self)

        # sequential model.
        self.model = Sequential(
            [
                # 128.
                Conv2D(filters=N_FILTER_BASE,
                    kernel_size=(KERNEL_CONV_SIZE, KERNEL_CONV_SIZE), strides=1,
                    activation='relu', padding='same',
                    input_shape=img_size + (1,)),
                AveragePooling2D(pool_size=(KERNEL_POOL_SIZE, KERNEL_POOL_SIZE),
                                 strides=STRIDE_POOL),

                # 64
                Conv2D(filters=2*N_FILTER_BASE,
                    kernel_size=(KERNEL_CONV_SIZE, KERNEL_CONV_SIZE), strides=1,
                    activation='relu', padding='same'),
                AveragePooling2D(pool_size=(KERNEL_POOL_SIZE, KERNEL_POOL_SIZE),
                                 strides=STRIDE_POOL),

                # Output.
                Flatten(),
                Dense(1, activation='sigmoid'),
            ]
        )


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Return predicted CNN output.
        """
        return self.model(inputs)