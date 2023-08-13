# Custom metrics class.
import sys
sys.path.insert(1, '../')

# get parameters.
from dev_modules.vcs_params import params_dataset
from dev_modules.vcs_params import params_train
from dev_modules.vcs_params import params_lite
from dev_modules.vcs_model import model_class
from dev_modules.vcs_params import params_model

from src.Logger import Logger

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
                 logger: Logger,
                 decision_threshold: float) -> None:
        """
        Define model and logger to obtain metrics.
        """
        self.logger = logger
        self.decision_threshold = decision_threshold


    def draw_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              title: str,
                              model_type: str) -> np.ndarray:
        """
        Plot the confusion matrix.
        Return the confusion matrix (skleanr standard).
        """
        y_pred = (y_pred >= self.decision_threshold).reshape(y_true.shape)
        y_pred = np.array(y_pred, dtype=int)
        
        # Calculate metrics.
        recall = sklearn.metrics.recall_score(y_true=y_true,
                                              y_pred=y_pred)
        acc = sklearn.metrics.accuracy_score(y_true=y_true,
                                              y_pred=y_pred)
        precision = sklearn.metrics.precision_score(y_true=y_true,
                                                    y_pred=y_pred)
        metrics_dict = {"{}_{}_recall".format(model_type, title): recall,
                        "{}_{}_acc".format(model_type, title): acc,
                        "{}_{}_precision".format(model_type, title): precision}
        self.logger.log_metrics(metrics_dict)
        print("{}_{}_Metrics\n".format(model_type, title), metrics_dict)

        # plot confusion matrix
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_true,
                                                            y_pred=y_pred)
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
        
        self.logger.log_plt_image(
            matrix_figure,
            "{}_{}_confusion_matrix".format(model_type, title))

        self.logger.log_artifact_pkl(
            confusion_matrix,
            "{}_{}_confusion_matrix.pkl".format(model_type, title))

        return confusion_matrix


    def visualize_errors(self,
                         title: str,
                         model_type: str,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         filepaths: str,
                         draw_errors: bool = False) -> list:
        """
        Save wrong inferences.
        Return name of wrong inferred archives.
        """
        y_pred = y_pred >= params_model.THRESHOLD_DECISION
        diff = y_pred - y_true
        # 0 : ok
        # 1 : pred==1, gt==0 -> FP
        # -1: pred==0, gt==1 -> FN
        
        errors_idx = np.nonzero(diff)[0]
        wrong_inferences = filepaths[errors_idx].tolist()
        
        # save images with wrong inferences.
        if draw_errors:
            for error in wrong_inferences:
                filename = model_type + "_" + title + "_" + error.split("/")[-1]
                sample = cv2.imread(error, cv2.IMREAD_GRAYSCALE)
                
                self.logger.log_cv2_image(sample, filename)
        
        self.logger.log_artifact_pkl(wrong_inferences,
                                     "{}_{}_errors.pkl".format(model_type,
                                                               title))
        return wrong_inferences


    def qt_metrics(self, y_fp: np.ndarray, y_qt: np.ndarray) -> list:
        """
        Get quantization error for fp and qt models.
        Return [quantization_errors_list, mean, std_deviation].
        """
        # signal errors.
        diff = y_qt - y_fp
        mean = np.mean(diff)
        std = np.std(diff)

        # abs errors.
        abs_diff = np.abs(diff)
        abs_mean = np.mean(np.abs(diff))
        abs_std = np.std(abs_diff)
        
        # logger metrics.
        metrics_dict = {"qt_mean": mean,
                        "qt_std": std,
                        "qt_abs_mean": abs_mean,
                        "qt_abs_std": abs_std}
        self.logger.log_metrics(metrics_dict)
        print("qt_Metrics\n", metrics_dict)
        
        # complete metrics.
        qt_metrics = dict(diff=diff, mean=mean, std=std,
                          abs_diff=abs_diff, abs_mean=abs_mean, abs_std=abs_std)
        
        self.logger.log_artifact_pkl(qt_metrics,
                                     "qt_metrics.pkl")
        
        return qt_metrics
