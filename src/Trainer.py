# Training routine.
import sys
sys.path.insert(1, '../')

# get parameters.
from dev_modules.vcs_params import params_dataset
from dev_modules.vcs_params import params_model
from dev_modules.vcs_params import params_train
# from dev_modules.vcs_params import params_lite

# import custom model.
from dev_modules.vcs_model import model_class

# logger.
from src.Logger import Logger
from src.Metrics import Custom_metrics
from src.Lite_handle import Lite_handler

# confere data.
import os

# data visualization.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tensor calculus backend.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# neural networks abstractions.
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, \
                                    Flatten, \
                                    Dense, \
                                    Dropout, \
                                    AveragePooling2D

# monitoring metrics and callbacks.
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

class Trainer():
    """
    Tensorflow train pipeline: floating point and quantized (lite).
    """
    
    def __init__(self) -> None:
        # reproducibility.
        self.logger = Logger()
        self.lite_h = Lite_handler()
        tf.keras.utils.set_random_seed(params_train.RANDOM_SEED)
        
    def end_run(self) -> None:
        self.logger.finish_run()


    def build_fp_model(self) -> tf.keras.models.Sequential:
        """
        Build floating point tensorflow model.
        Return sequential model created.
        """
        self.fp_model_h = model_class.fp_CNN_MCU()
        self.fp_model = self.fp_model_h.get_model()
        # define metrics to log.
        self.fp_model.compile(
            optimizer=keras.optimizers.Adam(params_train.LEARNING_RATE),
            loss=keras.losses.binary_crossentropy,
            metrics=[# critical for success.
                    keras.metrics.Recall(),
                    keras.metrics.BinaryAccuracy(),
                    keras.metrics.Precision(),
                    # auxiliar.
                    keras.metrics.FalseNegatives(),
                    keras.metrics.FalsePositives()],
        )
        self.fp_model.build((1, *params_dataset.IMAGE_SIZE))
        self.fp_model.summary(expand_nested=True)
        
        # return built model.
        return self.fp_model


    def train_val_split(self, augment:bool = False) -> list:
        """
        Tensorflow training generator.
        Return list [train_set, val_set].
        """
        dir_train = os.path.join("..",
                                params_dataset.DATASET_ROOT_DIR,
                                params_dataset.DATASET_TYPE,
                                params_dataset.TRAIN_DIR)

        if augment:
            gen_aug_option = params_train.TRAINING_AUG_GEN_PARAMS
            save_name = "aug"
            shuffle = True
        else:
            gen_aug_option = dict()
            save_name = "results"
            shuffle = False

        train_gen = ImageDataGenerator(**gen_aug_option,
                                       validation_split=params_train.PERC_VAL)

        self.train_set = train_gen.flow_from_directory(
            directory=dir_train,
            subset='training',
            shuffle=shuffle,
            **params_train.TRAINING_FLOW_PARAMS)
        print('Training set:\n', self.train_set.class_indices)
        train_files = self.train_set._filepaths

        self.val_set = train_gen.flow_from_directory(
            directory=dir_train,
            subset='validation',
            shuffle=shuffle,
            **params_train.TRAINING_FLOW_PARAMS)
        print('\nValidation set:\n', self.val_set.class_indices)
        val_files = self.val_set._filepaths

        # save file names for sets created.
        for sets_files in zip(["Train_files_{}.pkl".format(save_name),
                               "Validation_files_{}.pkl".format(save_name)],
                              [train_files, val_files]):
            self.logger.log_artifact_pkl(sets_files[1], sets_files[0])

        # return iterable sets.        
        return [self.train_set, self.val_set]


    def training_loop(self) -> dict:
        """
        Training loop for fp model.
        Return the training metrics history dict.
        """
        self.fp_models_dir = self.logger.create_model_dir("fp")
        tensorboard_dir = self.logger.create_tensorboard_dir()
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(self.fp_models_dir, "last")
                                            + params_train.MODEL_EXTENSION),
            keras.callbacks.ModelCheckpoint(os.path.join(self.fp_models_dir, "best")
                                            + params_train.MODEL_EXTENSION,
                                            **params_train.SAVE_BEST_PARAMS),
            keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
        ]
        self.fp_history = self.fp_model.fit(self.train_set,
                                 validation_data=self.val_set,
                                 callbacks=callbacks,
                                 **params_train.FIT_PARAMS,
                                 verbose=2)

        # log history metrics.
        self.logger.log_artifact_pkl(self.fp_history.history,
                                     params_train.HISTORY_FILE)
        
        # return training metric history dict. 
        return self.fp_history


    def load_model_trained(self) -> None:
        """
        Load last model trained.
        """
        best_model = os.path.join(self.fp_models_dir, "best") \
                                  + params_train.MODEL_EXTENSION
        self.fp_model = self.fp_model_h.load_model(best_model)


    def get_ground_truth(self,
                         set: ImageDataGenerator) -> tf.Tensor:
        """
        Abstraction to get "set" ground truths.
        """
        # if shuffle.
        if set.index_array is not None:
            ground_truth = set.classes[set.index_array]
        else:
            ground_truth = set.labels
        return ground_truth


    def init_metrics_handler(self) -> None:
        self.metrics_h = Custom_metrics(self.logger,
                                        params_model.THRESHOLD_DECISION)


    def get_confusion_matrix(self,
                             dataset: ImageDataGenerator,
                             title: str,
                             model_type: str):
        """
        Obtain confusion matrix for set given its ground truths.
        """
        y_ground_truth = self.get_ground_truth(dataset)
        
        if model_type == "fp":
            y_preds = self.fp_model_h.fp_predict(dataset)
        elif model_type == "qt":
            y_preds = self.qt_model_h.qt_predict(dataset)
        
        confusion_matrix = self.metrics_h.draw_confusion_matrix(y_ground_truth,
                                                                y_preds,
                                                                title,
                                                                model_type)
        print("Confusion matrix", title, '\n', confusion_matrix)


    def get_errors(self,
                   dataset: ImageDataGenerator,
                   title: str,
                   model_type: str,
                   draw_errors: bool = False):
        """
        Obtain wrong inferences for given set.
        """
        y_ground_truth = self.get_ground_truth(dataset)
        
        if model_type == "fp":
            y_preds = self.fp_model_h.fp_predict(dataset)
        elif model_type == "qt":
            y_preds = self.qt_model_h.qt_predict(dataset)
        
        y_preds = y_preds.reshape(y_ground_truth.shape)
        filepaths = np.array(dataset.filepaths)
        errors_list = self.metrics_h.visualize_errors(title,
                                                      model_type,
                                                      y_ground_truth, y_preds,
                                                      filepaths,
                                                      draw_errors=draw_errors)
        print("Errors list", title, '\n', errors_list)


    def build_qt_model(self) -> tf.keras.models.Sequential:
        """
        Build quantized tensorflow model.
        Return sequential model created.
        """
        self.qt_model_dir = self.logger.create_model_dir("qt")
        self.qt_model_h = model_class.qt_CNN_MCU()
        self.qt_model =  self.qt_model_h.get_model(self.fp_model,
                                                   self.train_set,
                                                   self.qt_model_dir)
        
        # return built model.
        return self.qt_model


    def quantization_error(self, dataset: ImageDataGenerator) -> list:
        """
        Get quantization error for fp and qt models.
        Return [quantization_errors_list, mean, std_deviation].
        """
        y_fp = self.fp_model_h.fp_predict(dataset).flatten()
        y_qt = self.qt_model_h.qt_predict(dataset).flatten()
        qt_metrics = self.metrics_h.qt_metrics(y_fp, y_qt)
        
        # return qt errors.
        return qt_metrics

    def test_set_gen(self) -> list:
        """
        Tensorflow test generator.
        Return test_set.
        """
        dir_test = os.path.join("..",
                                params_dataset.DATASET_ROOT_DIR,
                                params_dataset.DATASET_TYPE,
                                params_dataset.TEST_DIR)

        test_gen = ImageDataGenerator()

        self.test_set = test_gen.flow_from_directory(
            directory=dir_test,
            shuffle=False,
            **params_train.TRAINING_FLOW_PARAMS)
        print('Test set:\n', self.test_set.class_indices)
        test_files = self.test_set._filepaths

        # save file names for test set.
        self.logger.log_artifact_pkl(test_files, "Test_files.pkl")

        # return iterable sets.        
        return self.test_set