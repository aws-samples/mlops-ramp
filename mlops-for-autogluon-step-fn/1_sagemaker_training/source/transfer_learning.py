import argparse
import json
import logging
import os
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from constants import constants


logging.basicConfig(level=logging.WARNING)  # Use logging.WARNING since logging.INFO is ignored by AutoGluon DLC


def prepare_data(data_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Read data from train and validation channel, and return predicting features and target variables.

    If data.csv under validation channel is not found, validation data will be None.

    Args:
        data_dir (str): directory which saves the training data.

    Returns:
        Tuple of training features, training target, validation features, validation target.
    """
    print(os.path.join(data_dir, constants.INPUT_DATA_FILENAME))
    df_train = pd.read_csv(
        os.path.join(data_dir, constants.INPUT_DATA_FILENAME)
    )
#     df_train.columns = ["target"] + [f"feature_{x}" for x in range(df_train.shape[1] - 1)]

#     try:
#         df_validation = pd.read_csv(
#             os.path.join(data_dir, constants.VALIDATION_CHANNEL, constants.INPUT_DATA_FILENAME),
#             header=None,
#         )
#         df_validation.columns = ["target"] + [f"feature_{x}" for x in range(df_validation.shape[1] - 1)]

#     except FileNotFoundError:  # when validation data is not available in the directory
#         logging.warning(
#             "Validation data is not found. A certain fraction of training data is "
#             "randomly selected as validation data. The fraction value is selected based on the "
#             "number of rows in the training data. For details, see the official documentation: "
#             "https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.fit"
#         )
#         df_validation = None

    return df_train
# , df_validation


def _parse_args():
    parser = argparse.ArgumentParser()
    
    

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--pretrained-model", type=str, default=os.environ.get("SM_CHANNEL_MODEL"))
    parser.add_argument("--n_gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--eval_metric", type=str, default=constants.DEFAULT_EVAL_METRIC)
    parser.add_argument("--presets", type=str, default=constants.DEFAULT_PRESETS)
    parser.add_argument("--auto_stack", type=str, default=constants.DEFAULT_AUTO_STACK)
    parser.add_argument("--num_bag_folds", type=str, default=constants.DEFAULT_NUM_BAG_FOLDS)
    parser.add_argument("--num_bag_sets", type=int, default=constants.DEFAULT_NUM_BAG_SETS)
    parser.add_argument("--num_stack_levels", type=int, default=constants.DEFAULT_NUM_STACK_LEVELS)
    parser.add_argument("--refit_full", type=str, default=constants.DEFAULT_REFIT_FULL)
    parser.add_argument("--set_best_to_refit_full", type=str, default=constants.DEFAULT_SET_BEST_TO_REFIT_FULL)
    parser.add_argument("--save_space", type=str, default=constants.DEFAULT_SAVE_SPACE)
    parser.add_argument("--verbosity", type=int, default=constants.DEFAULT_VERBOSITY)

    return parser.parse_known_args()


def run_with_args(args):
    """Run training."""
    print("==== printing args ====")
    print(args)
    print(os.path.join(args.train))
    df_train = prepare_data(data_dir=os.path.join(args.train))
    print(df_train.head())

    if args.n_gpus:
        logging.warning(f"Running training job with the number of gpu: {args.n_gpus}")

    ag_predictor_args = {
        constants.LABEL: constants.TARGET_COLUMN,
        constants.PATH: args.model_dir,
        constants.EVAL_METRIC: constants.EVALUATION_METRIC if args.eval_metric == "auto" else args.eval_metric,
        constants.PROBLEM_TYPE: constants.PROBLEM_TYPE_OBJECTIVE,
    }

    if int(args.num_bag_folds) == 0 and args.num_stack_levels != 0:
        logging.warning(
            f"Hyperparameter num_stack_levels is found as {args.num_stack_levels}. "
            f"However, num_stack_levels must be 0 if num_bag_folds is 0. "
            f"Overwriting num_stack_levels to be 0."
        )
        args.num_stack_levels = 0

    kwargs = {
        constants.AUTO_STACK: args.auto_stack == "True",
        constants.NUM_BAG_FOLDS: int(args.num_bag_folds),
        constants.NUM_BAG_SETS: args.num_bag_sets,
        constants.NUM_STACK_LEVELS: args.num_stack_levels,
        constants.REFIT_FULL: args.refit_full == "True",
        constants.SET_BEST_TO_REFIT_FULL: args.set_best_to_refit_full == "True",
        constants.SAVE_SPACE: args.save_space == "True",
        constants.VERBOSITY: args.verbosity,
    }

    TabularPredictor(**ag_predictor_args).fit(train_data=df_train, presets=args.presets, **kwargs)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
