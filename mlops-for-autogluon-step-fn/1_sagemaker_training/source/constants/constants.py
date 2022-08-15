DEPENDENCIES = []  # This is not used at the endpoint.
# It is used while launching the job to decide which all dependency packages
# are to be uploaded to the training container.

INPUT_DATA_FILENAME = "train.csv"
TRAIN_CHANNEL = "train"
VALIDATION_CHANNEL = "validation"

TARGET_COLUMN = "OT"

LABEL = "label"
PATH = "path"
EVAL_METRIC = "eval_metric"
PROBLEM_TYPE = "problem_type"
AUTO_STACK = "auto_stack"
NUM_BAG_FOLDS = "num_bag_folds"
NUM_BAG_SETS = "num_bag_sets"
NUM_STACK_LEVELS = "num_stack_levels"
REFIT_FULL = "refit_full"
SET_BEST_TO_REFIT_FULL = "set_best_to_refit_full"
SAVE_SPACE = "save_space"
VERBOSITY = "verbosity"

DEFAULT_EVAL_METRIC = "auto"
DEFAULT_PRESETS = "medium_quality"
DEFAULT_AUTO_STACK = "False"
DEFAULT_NUM_BAG_FOLDS = "0"
DEFAULT_NUM_BAG_SETS = 1
DEFAULT_NUM_STACK_LEVELS = 0
DEFAULT_REFIT_FULL = "False"
DEFAULT_SET_BEST_TO_REFIT_FULL = "False"
DEFAULT_SAVE_SPACE = "False"
DEFAULT_VERBOSITY = 2

PROBLEM_TYPE_OBJECTIVE = "regression"
EVALUATION_METRIC = "root_mean_squared_error"
