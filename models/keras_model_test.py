import json
import os
from pathlib import Path

import keras.layers
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from housing_model.evaluations.keras_model_evaluator import eval_model_on_tfds
from housing_model.models.keras_model import (
    ModelBuilder,
    HyperParams,
    bits_to_num,
    ArchitectureParams,
    KerasModelTrainer,
    ModelParams,
    TrainParams,
    ExperimentSpec,
)
from housing_model.models.trainer import train_job


def test_bits_to_num():
    bits = tf.constant([[1, 0, 1], [0, 1, 0]], dtype="float32")
    res = bits_to_num(bits, 3)
    np.testing.assert_almost_equal(res.numpy(), [5, 2])


def test_model_builder_input_output():
    model_builder = ModelBuilder(
        ModelParams(HyperParams(embedding_size=5), ArchitectureParams({"f1", "f2"}))
    )

    model = model_builder.build()
    assert len(model.inputs) == 2
    assert len(model.outputs) == 2


def test_model_builder_generate_price():
    model_builder = ModelBuilder(
        ModelParams(HyperParams(embedding_size=5), ArchitectureParams({"f1", "f2"}))
    )

    model = model_builder.build()
    f1 = tf.constant([1.0])
    f2 = tf.constant([2.0])

    output = model({"f1": f1, "f2": f2})

    assert output is not None
    assert isinstance(output, dict)
    assert "sold_price" in output
    assert output["sold_price"].numpy().shape == (1,)

    assert "bits" in output
    assert output["bits"].numpy().shape == (1, 32)


def test_model_overfit():
    num_bits = 3
    model_builder = ModelBuilder(
        ModelParams(
            HyperParams(embedding_size=5),
            ArchitectureParams({"f1", "f2"}, num_bits=num_bits),
        )
    )

    model = model_builder.build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-1, clipnorm=1.0),
        loss={
            # 'sold_price': 'mean_squared_error',
            "bits": "binary_crossentropy"
        },
    )

    f1 = tf.constant([1.0, 2.0])
    f2 = tf.constant([2.0, 3.0])
    sold_prices = tf.constant([3.0, 4.0])
    bits = tf.constant([[0, 1, 1], [1, 0, 0]])

    inputs = {"f1": f1, "f2": f2}

    hist = model.fit(x=inputs, y={"sold_price": sold_prices, "bits": bits}, epochs=1000)

    assert hist.history["loss"][-1] < 1e-3


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
    for stored, loaded in zip(learned_model.weights, loaded_model.weights):
        diff = stored - loaded
        assert all(-0.0001 < diff < 0.0001)
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
    config_dir = f"{test_dir}/../"
    experiment_config_file = f"{config_dir}/experiment.json"

    with open(experiment_config_file) as fin:
        experiment_config = ExperimentSpec.from_json(fin.read())

    # check the model architecture does not have any error
    check_model_architecture(experiment_config, Path(str(tmpdir)), train_ds)


def test_save_load(tmpdir):
    tmpdir = Path(str(tmpdir))
    test_dir = os.path.dirname(os.path.realpath(__file__))

    config_dir = f"{test_dir}/../"
    experiment_config_file = f"{config_dir}/experiment.json"

    with open(experiment_config_file) as fin:
        experiment_config = ExperimentSpec.from_json(fin.read())
        expected_trainer = KerasModelTrainer.build(experiment_config.modeling, tmpdir)
        # experiment_config.training.epochs = 1
        # train_job(experiment_config, str(tmpdir))

    expected_trainer.save()
    actual_trainer = KerasModelTrainer.load(tmpdir)

    for actual_layer, expected_layer in zip(
        actual_trainer.keras_model.layers, expected_trainer.keras_model.layers
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
