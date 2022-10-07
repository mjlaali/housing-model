import json
import os
from collections import OrderedDict
from pprint import pprint

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from housing_model.evaluations.keras_model_evaluator import eval_model_on_tfds
from housing_model.models import trainer
from housing_model.models.keras_model import (
    ModelBuilder,
    HyperParams,
    bits_to_num,
    ArchitectureParams,
    KerasModelTrainer,
    ModelParams,
    TrainParams,
)


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
    model_params: ModelParams,
    model_path: str,
    train_ds: tf.data.Dataset,
    overfit_train_params: TrainParams,
):
    ex_cnt = overfit_train_params.batch_size
    train_ds = train_ds.take(ex_cnt).cache()
    test_ds = train_ds.take(ex_cnt).cache()

    keras_model = KerasModelTrainer.build(model_params)

    overfit_loss = get_overfit_loss(train_ds, keras_model, overfit_train_params)

    keras_model.save(model_path)
    # keras_model = KerasModelTrainer.load(model_path)
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
    train_ds = tfds.load("tf_housing", split="train").take(4).cache()
    test_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = f"{test_dir}/../"
    model_params_path = f"{config_dir}/model_params.json"
    overfit_train_params_path = f"{config_dir}/overfit_train_params.json"

    with open(model_params_path) as fin:
        model_params = ModelParams.from_json(fin.read())

    with open(overfit_train_params_path) as fin:
        overfit_train_params = TrainParams.from_json(fin.read())

    # check the model architecture does not have any error
    check_model_architecture(model_params, tmpdir, train_ds, overfit_train_params)
