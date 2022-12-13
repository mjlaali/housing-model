import json
import os
from datetime import datetime
from pathlib import Path

import keras.layers
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.utils import io_utils
from keras.utils import tf_utils

from housing_model.evaluations.keras_model_evaluator import eval_model_on_tfds
from housing_model.modeling.naive_deep.configs import (
    DatasetSpec,
    TrainParams,
)
from housing_model.modeling.naive_deep.model_trainer import (
    KerasModelTrainer,
    AdaptiveLoss,
)
from housing_model.training.trainer import train_job, ExperimentSpec


class TerminateOnAlmostZeroLoss(tf.keras.callbacks.Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self, almost_zero, metric_key="loss"):
        super().__init__()
        self._metric_key = metric_key
        self._almost_zero = almost_zero

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get(self._metric_key)
        if loss is not None:
            loss = tf_utils.sync_to_numpy_or_python_type(loss)
            if loss <= self._almost_zero:
                io_utils.print_msg(
                    f"Batch {batch}: loss is almost zero"
                )
                self.model.stop_training = True


def get_overfit_loss(
    train_ds: tf.data.Dataset,
    keras_model: KerasModelTrainer,
    overfit_train_params: TrainParams,
) -> float:
    dataset_size = len(list(train_ds))
    hist = keras_model.fit_model(
        train_ds,
        train_ds.take(dataset_size).cache(),
        overfit_train_params,
        additional_callbacks=[TerminateOnAlmostZeroLoss(almost_zero=0.0001)]
    )
    return hist.history["loss"][-1]


def check_model_architecture(
    experiment_config: ExperimentSpec,
    model_path: Path,
    train_ds: tf.data.Dataset,
):
    ex_cnt = experiment_config.training.batch_size
    train_ds = train_ds.take(ex_cnt).cache()
    test_ds = train_ds.take(ex_cnt).cache()

    keras_model = KerasModelTrainer.build(experiment_config.modeling, model_path)

    overfit_loss = get_overfit_loss(train_ds, keras_model, experiment_config.training)

    learned_model = keras_model.keras_model
    keras_model.save()
    keras_model = KerasModelTrainer.load(model_path)
    loaded_model = keras_model.keras_model

    keras_models_are_almost_equal(learned_model, loaded_model)

    predictor = keras_model.make_predictor()

    metric = eval_model_on_tfds(test_ds, predictor)
    print(json.dumps(metric.value, indent=2, sort_keys=True))
    assert (
        metric.value["mean"] < 0.01
    ), f"The mean percentage error ({metric.value['mean']}) is too high"
    assert (
        overfit_loss < 1e-3
    ), f"The model did not overfit! loss ({overfit_loss}) is too high"


def test_overfit(tmpdir):
    train_ds = tfds.load("tf_housing", split="201912").take(4).cache()
    test_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = f"{test_dir}/../../"
    experiment_config_file = f"{config_dir}/experiment.json"

    with open(experiment_config_file) as fin:
        experiment_config = ExperimentSpec.from_json(fin.read())

    # check the model architecture does not have any error
    check_model_architecture(experiment_config, Path(str(tmpdir)), train_ds)


def test_save_load(tmpdir):
    tmpdir = Path(str(tmpdir))
    test_dir = os.path.dirname(os.path.realpath(__file__))

    config_dir = f"{test_dir}/../../"
    experiment_config_file = f"{config_dir}/experiment.json"

    with open(experiment_config_file) as fin:
        experiment_config = ExperimentSpec.from_json(fin.read())
        expected_trainer = KerasModelTrainer.build(experiment_config.modeling, tmpdir)
        experiment_config.training.epochs = 1
        experiment_config.datasets.train = DatasetSpec(
            datetime(2019, 1, 1), datetime(2019, 2, 1)
        )
        train_job(experiment_config, str(tmpdir))

    expected_trainer.save()
    actual_trainer = KerasModelTrainer.load(tmpdir)

    keras_models_are_almost_equal(actual_trainer.keras_model, expected_trainer.keras_model)


def keras_models_are_almost_equal(actual_model: tf.keras.models.Model, expected_model: tf.keras.models.Model):
    for actual_layer, expected_layer in zip(
            actual_model.layers, expected_model.layers
    ):
        if not isinstance(actual_layer, keras.layers.InputLayer):
            for w_actual, w_expected in zip(
                    actual_layer.get_weights(),
                    expected_layer.get_weights(),
            ):
                np.testing.assert_almost_equal(
                    w_actual,
                    w_expected,
                    err_msg=f"{actual_layer.name} and {expected_layer.name}",
                )


def test_adaptive_loss():
    a_loss = AdaptiveLoss("mse", lambda epoch, w_init: w_init**epoch, 0.5)
    config = a_loss.get_config()
    json_str = json.dumps(config)
    loaded_loss = AdaptiveLoss.from_config(json.loads(json_str))

    manual = 0.5**1 * tf.keras.losses.get("mse")(np.asarray([1.0]), np.asarray([2.0]))
    expected = a_loss(np.asarray([1.0]), np.asarray([2.0]))
    np.testing.assert_almost_equal(manual, expected)

    actual = loaded_loss(np.asarray([1.0]), np.asarray([2.0]))
    np.testing.assert_almost_equal(expected, actual)
