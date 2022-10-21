import argparse
import json
import logging
import shutil

import tensorflow as tf
import tensorflow_datasets as tfds

from housing_model.evaluations.keras_model_evaluator import eval_model_on_tfds
from housing_model.models.keras_model import KerasModelTrainer, TrainParams, ModelParams

logger = logging.getLogger(__name__)


def main(
    model_params_path: str,
    model_path: str,
    train_params_path: str,
):
    logger.info(
        "Please first run keras_model_test before running trainer.py to make sure the model get trained."
    )

    train_ds: tf.data.Dataset = tfds.load("tf_housing", split="train")

    with open(model_params_path) as fin:
        model_params = ModelParams.from_json(fin.read())

    # start training job and export the model
    dev_ds = tfds.load("tf_housing", split="dev")
    keras_model = KerasModelTrainer.build(model_params)
    with open(train_params_path) as fin:
        train_params = TrainParams.from_json(fin.read())

    keras_model.fit_model(
        train_ds.shuffle(train_params.batch_size * 1000, reshuffle_each_iteration=True),
        dev_ds,
        train_params,
    )

    shutil.rmtree(model_path, ignore_errors=True)
    keras_model.save(model_path)

    # test the exported model
    test_ds = tfds.load("tf_housing", split="test")
    keras_model = KerasModelTrainer.load(model_path)
    predictor = keras_model.make_predictor()
    metrics = eval_model_on_tfds(test_ds, predictor)
    print(json.dumps(metrics.value, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_params_path", required=True)
    parser.add_argument("--train_params_path", required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    main(**vars(args))
