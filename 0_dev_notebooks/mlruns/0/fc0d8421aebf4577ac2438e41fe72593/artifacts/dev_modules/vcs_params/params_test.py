# Test hyperparameters.
import datetime
import os
from dev_modules.vcs_params import params_dataset


RANDOM_SEED = 0

# experiment run infos.
RUN_NAME = "test_0"
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + RUN_NAME
TENSORFLOW_DIR = "./logs/" + RUN_NAME

# data information.
CLASSES = {'0_utilizavel': 0,
           '1_defeituoso': 1}

# dataloader options.
BATCH_SIZE = 16
TEST_FLOW_PARAMS = dict(target_size=params_dataset.IMAGE_SIZE,
                        color_mode='grayscale',
                        classes=CLASSES,
                        class_mode='binary',
                        batch_size=BATCH_SIZE,
                        seed=RANDOM_SEED)

# dirs to save.
TEST_OUTPUTS_DIR = os.path.join("test_outputs", RUN_NAME)
TEST_ERRORS_DIR = "test_errors"

# metrics files.
CONFUSION_TEST_FILE = "confusion_matrix_test.pkl"
WRONG_TEST_FILE = "wrong_train_files.pkl"
