import argparse
import datetime
import json
import logging
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
import tensorflow_datasets as tfds

from housing_model.evaluations.keras_model_evaluator import eval_model_on_tfds
from housing_model.models.keras_model import (
    KerasModelTrainer,
    ExperimentSpec,
    DatasetSpec,
)

logger = logging.getLogger(__name__)


def create_dataset(
    datasets: Dict[str, tf.data.Dataset], dataset_spec: DatasetSpec
) -> tf.data.Dataset:
    a_date = dataset_spec.start
    selected: List[tf.data.Dataset] = []

    while a_date < dataset_spec.end:
        a_date += timedelta(days=31)
        # reset the day to the first day of month
        a_date = datetime.datetime(a_date.year, a_date.month, 1)
        a_dataset = datasets.get(a_date.strftime("%Y%m"))
        if a_dataset:
            selected.append(a_dataset)

    choice_dataset = tf.data.Dataset.range(len(selected)).repeat()
    final_dataset = tf.data.Dataset.choose_from_datasets(selected, choice_dataset)

    return final_dataset


def main(
    experiment: str,
    output: str,
):
    logger.info(
        "Please first run keras_model_test before running trainer.py to make sure the model get trained."
    )

    shutil.rmtree(output, ignore_errors=True)
    output_path = Path(output)
    output_path.mkdir(parents=True)
    shutil.copy(experiment, output_path / "exp_config.json")

    with open(experiment) as fin:
        experiment_spec = ExperimentSpec.from_json(fin.read())

    datasets: Dict[str, tf.data.Dataset] = tfds.load("tf_housing")

    keras_model = KerasModelTrainer.build(experiment_spec.modeling)
    train_ds = create_dataset(datasets, experiment_spec.datasets.train)
    dev_ds = create_dataset(datasets, experiment_spec.datasets.dev)

    keras_model.fit_model(
        train_ds.shuffle(
            experiment_spec.training.batch_size * 1000, reshuffle_each_iteration=True
        ),
        dev_ds,
        experiment_spec.training,
    )

    keras_model.save(output)

    # test the exported model
    test_ds = create_dataset(datasets, experiment_spec.datasets.test)
    keras_model = KerasModelTrainer.load(output)
    predictor = keras_model.make_predictor()
    metrics = eval_model_on_tfds(test_ds, predictor)

    with open(output_path / "eval.json", "w") as eval_file:
        eval_json = json.dumps(metrics.value, indent=2, sort_keys=True)
        eval_file.write(eval_json)
        print(eval_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(**vars(args))
