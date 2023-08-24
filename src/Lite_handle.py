# Class to deal with tensorflow lite specificities.
import sys
import os
sys.path.insert(1, '../')

# get parameters.
from dev_modules.vcs_params import params_lite

# logger.
from src.Logger import Logger

# numeric dealing.
import numpy as np
import cv2

# tensor calculus backend.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Lite_handler():
    """
    Class to abstract use of tensorflow lite.
    """
    def representative_dataset_gen(self) -> np.ndarray:
        """
        Provide representative dataset to quantization.
        """
        for path in self.dataset._filepaths:
            sample = np.float32(cv2.imread(path, cv2.IMREAD_GRAYSCALE) )
            yield([sample.reshape(-1, *sample.shape, 1)])


    def load_model(self, qt_model: bytes) -> None:
        """
        Load quantized model.
        """
        self.qt_model = qt_model


    def build_quantized_model(self,
                              fp_model: tf.keras.models.Sequential,
                              dataset: ImageDataGenerator,
                              save_dir: str) -> tf.keras.models.Sequential:
        """
        Build quantized model from floating point.
        """
        self.dataset = dataset
        converter = tf.lite.TFLiteConverter.from_keras_model(fp_model)

        # Enforce integer 8 bits quantization.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Create lite model.
        converter.representative_dataset = self.representative_dataset_gen
        self.qt_model = converter.convert()

        # Save the model in disk.
        model_file = os.path.join(save_dir, "qt_model.tflite")
        open(model_file, "wb").write(self.qt_model)
        
        # assign interpreter to model.
        self.interpreter = tf.lite.Interpreter(model_content=self.qt_model)
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        print('QT model detais:')
        print('input details =', input_details)
        print('output details =', output_details)
        return self.qt_model


    def predict_tflite(self, x_test: np.ndarray) -> np.ndarray:
        """
        Routine to do inferences in tflite model.
        Return the predictions.
        """
        x_input_ = x_test.copy()
        x_input_ = x_input_
        x_input_ = x_input_.astype(np.float32)

        # Initialize the TFLite interpreter.
        self.interpreter = tf.lite.Interpreter(model_content=self.qt_model)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        # Scale and quantize the input layer.
        input_scale, input_zero_point = input_details["quantization"]
        if (input_scale, input_zero_point) != (0.0, 0):
            x_input_ = x_input_ / input_scale + input_zero_point
            x_input_ = x_input_.astype(input_details["dtype"])
        
        # Invoke the interpreter.
        y_pred = np.empty(x_input_.size, dtype=output_details["dtype"])
        for i in range(len(x_input_)):
            self.interpreter.set_tensor(input_details["index"], [x_input_[i]])
            self.interpreter.invoke()
            y_pred[i] = self.interpreter.get_tensor(output_details["index"])[0]
        
        # Dequantize the output layer.
        output_scale, output_zero_point = output_details["quantization"]
        if (output_scale, output_zero_point) != (0.0, 0):
            y_pred = y_pred.astype(np.float32)
            y_pred = (y_pred - output_zero_point) * output_scale

        return y_pred
