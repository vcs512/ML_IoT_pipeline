# Training hyperparameters.
import datetime
import os
from dev_modules.vcs_params import params_dataset


RANDOM_SEED = 0

# experiment run infos.
RUN_NAME = "half_trunc_40epoch"
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + RUN_NAME
TENSORBOARD_DIR = "./logs/" + RUN_NAME

# data information.
CLASSES = {'0_utilizavel': 0,
           '1_defeituoso': 1}

# dataloader options.
TRAIN_FILES = "train_files.pkl"
VAL_FILES = "val_files.pkl"
BATCH_SIZE = 16
PERC_VAL = 0.20
TRAINING_AUG_GEN_PARAMS = dict(horizontal_flip=True,
                               vertical_flip=True)
TRAINING_FLOW_PARAMS = dict(target_size=params_dataset.IMAGE_SIZE,
                            color_mode='grayscale',
                            classes=CLASSES,
                            class_mode='binary',
                            batch_size=BATCH_SIZE,
                            seed=RANDOM_SEED)

# saved model options.
MODEL_EXTENSION = ".h5"
SAVE_BEST_PARAMS = dict(monitor="val_loss", save_best_only=True)

# dirs to save.
TRAIN_OUTPUTS_DIR = os.path.join("train_outputs", RUN_NAME)
MODELS_DIR = "models"
TRAIN_ERRORS_DIR = "train_errors"
VAL_ERRORS_DIR = "validation_errors"

# metrics files.
HISTORY_FILE = "history.pkl"
CONFUSION_TRAIN_FILE = "confusion_matrix_train.pkl"
CONFUSION_VAL_FILE = "confusion_matrix_val.pkl"
WRONG_TRAIN_FILE = "wrong_train_files.pkl"
WRONG_VAL_FILE = "wrong_val_files.pkl"

# training loop hyperparameters.
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
FIT_PARAMS = dict(batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS)
