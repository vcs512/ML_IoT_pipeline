# Custom metrics class.
import sys
sys.path.insert(1, '../')

# get parameters.
from dev_modules.vcs_params import params_dataset
from dev_modules.vcs_params import params_train
from dev_modules.vcs_params import params_lite
from dev_modules.vcs_model import model_class
from dev_modules.vcs_params import params_model

from Logger import Logger

import os
import natsort
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import sklearn
import tensorflow as tf


class Custom_metrics():
    """
    Class to abstract use of usual metrics.
    """
    def __init__(self,
                 model: model_class.fp_CNN_MCU,
                 logger: Logger,
                 decision_threshold: float) -> None:
        """
        Define model and logger to obtain metrics.
        """
        self.model = model
        self.logger = logger
        self.decision_threshold = decision_threshold


    def draw_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              title: str) -> np.ndarray:
        """
        Plot the confusion matrix.
        Return the confusion matrix (skleanr standard).
        """
        y_pred = (y_pred >= self.decision_threshold).reshape(y_true.shape)
        y_pred = np.array(y_pred, dtype=int)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_true,
                                                            y_pred=y_pred)

        # plot confusion matrix
        matrix_figure = plt.figure()
        ax = sns.heatmap(confusion_matrix,
                         annot=True,
                         annot_kws={'size':14, 'weight':'bold'},
                         fmt='d', cbar=False, cmap='Blues')

        plt.title(title, size=25, weight='bold')
        CLASSES_LIST = ['Usable', 'Defective']
        plt.xlabel('Predicted', size=14, weight='bold')
        ax.set_xticklabels(CLASSES_LIST)
        plt.ylabel('Ground truth', size=14, weight='bold')
        ax.set_yticklabels(CLASSES_LIST, va='center')
        plt.tick_params(axis='both', labelsize=14, length=0)
        
        self.logger.log_plt_image(matrix_figure,
                                  "{}_confusion_matrix".format(title))

        self.logger.log_artifact_pkl(confusion_matrix,
                                     "{}_confusion_matrix.pkl".format(title))

        return confusion_matrix


    def visualize_errors(self,
                         title: str,
                         files_set: list) -> list:
        """
        Save wrong inferences.
        Return name of wrong inferred archives.
        """
        wrong_inference = list()

        for file in files_set:
            ground_truth = int(file.split('/')[-2][0])
            filename = title + "_" + file.split("/")[-1]

            sample = cv2.imread(file, cv2.IMREAD_GRAYSCALE).reshape(
                -1,
                params_dataset.IMAGE_SIZE[0], params_dataset.IMAGE_SIZE[1],
                1)
            pred = self.model.fp_predict(sample)
            pred = (pred >= params_model.THRESHOLD_DECISION).reshape(-1,)
            
            diff = pred - ground_truth
            # 0 : ok
            # 1 : pred==1, gt==0 -> FP
            # -1: pred==0, gt==1 -> FN

            if diff:
                wrong_inference.append(filename)
                image = sample[0, :, :, 0]
                self.logger.log_cv2_image(image, filename)
        
        wrong_inference = natsort.natsorted(wrong_inference)
        self.logger.log_artifact_pkl(wrong_inference,
                                     "{}_errors.pkl".format(title))
        return wrong_inference
