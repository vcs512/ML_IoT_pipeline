# Test hyperparameters.
import datetime
import os
from dev_modules.vcs_params import params_dataset


RANDOM_SEED = 0

# training infos.
TRAIN_RUN_NAME = "20230807_201238_old"
TRAIN_RUN_DIR = os.path.join("..", "3_train", "train_outputs", TRAIN_RUN_NAME)
TRAINED_MODEL = os.path.join(TRAIN_RUN_DIR, "checkpoints", "best.h5")

# lite conversion infos.
LITE_RUN_NAME = "test_0"
LITE_RUN_NAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + LITE_RUN_NAME
LITE_OUTPUTS_DIR = os.path.join("lite_outputs", LITE_RUN_NAME)
LITE_MODEL_DIR = os.path.join(LITE_OUTPUTS_DIR, "model")
LITE_ERRORS_DIR = os.path.join(LITE_OUTPUTS_DIR, "lite_errors")

# data information.
CLASSES = {'0_utilizavel': 0,
           '1_defeituoso': 1}

# dataloader options.
TRAIN_FILES = "train_files.pkl"
VAL_FILES = "val_files.pkl"
BATCH_SIZE = 16
VALIDATION_IN_TRAIN = 0.20
TRAINING_GEN_PARAMS = dict(validation_split=VALIDATION_IN_TRAIN)
TRAINING_FLOW_PARAMS = dict(target_size=params_dataset.IMAGE_SIZE,
                            color_mode='grayscale',
                            classes=CLASSES,
                            class_mode='binary',
                            batch_size=BATCH_SIZE,
                            seed=RANDOM_SEED)

# metrics files.
CONFUSION_TRAIN_FILE = "confusion_matrix_train.pkl"
CONFUSION_VAL_FILE = "confusion_matrix_val.pkl"

TRAIN_ERRORS_DIR = "train_errors"
WRONG_TRAIN_FILE = "wrong_train_files.pkl"

VAL_ERRORS_DIR = "validation_errors"
WRONG_VAL_FILE = "wrong_val_files.pkl"
