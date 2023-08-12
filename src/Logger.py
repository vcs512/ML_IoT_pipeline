# Logger class.
import sys
sys.path.insert(1, '../')

# get parameters.
from dev_modules.vcs_params import params_dataset
from dev_modules.vcs_params import params_model
from dev_modules.vcs_params import params_train
from dev_modules.vcs_params import params_lite

import mlflow
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Logger():
    """
    Abstraction to log metrics and parameters.
    """
    def __init__(self: str) -> None:
        """
        Initialize mlflow and local loggers.
        """
        mlflow.tensorflow.autolog()
        mlflow.start_run(run_name=params_train.RUN_NAME)
        # log model and training gen parameters.
        mlflow.log_artifact("../dev_modules/")
        mlflow.log_params(params_model.TRAIN_HYPER)
        # create local dirs.]
        self.output_dir = os.path.join('.', params_train.TRAIN_OUTPUTS_DIR)
        os.makedirs(self.output_dir)
        self.errors_dir = os.path.join(self.output_dir, params_train.TRAIN_ERRORS_DIR)
        os.makedirs(self.errors_dir)


    def log_artifact_pkl(self, object_to_save: object, name: str) -> None:
        """
        Log object_to_save as a 'name.pkl'.
        """
        with open(os.path.join(self.output_dir, name), 'wb') as file:
            pickle.dump(object_to_save, file)


    def create_model_dir(self, type: str) -> str:
        """
        Create checkpoints dir for model training.
        Return the models checkpoint dir created.
        """
        self.models_checkpoints_dir = os.path.join(self.output_dir, type,
                                                   params_train.MODELS_DIR)
        os.makedirs(self.models_checkpoints_dir)
        return self.models_checkpoints_dir


    def create_tensorboard_dir(self) -> str:
        """
        Create tensorboard dir for model training.
        Return the dir created.
        """
        self.tensorboard_dir = os.path.join('.',
                                            params_train.TENSORBOARD_DIR)
        os.makedirs(self.tensorboard_dir)
        return self.tensorboard_dir


    def log_plt_image(self, figure: plt.figure, name: str) -> None:
        """
        Save a matplotlib image.
        """
        save_dir = os.path.join(self.output_dir, name)
        figure.savefig(save_dir)

    
    def log_cv2_image(self, image: np.ndarray, name: str) -> None:
        """
        Save a matplotlib image.
        """
        save_dir = os.path.join(self.errors_dir, name)
        cv2.imwrite(save_dir, image)


    def finish_run(self):
        """
        Log all in outputs dir and end run in mlflow.
        """
        mlflow.log_artifact(self.output_dir)
        mlflow.end_run()
