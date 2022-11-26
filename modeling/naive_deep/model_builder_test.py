import numpy as np
import tensorflow as tf

from housing_model.modeling.naive_deep.configs import (
    HyperParams,
    ArchitectureParams,
    ModelParams,
)
from housing_model.modeling.naive_deep.model_builder import (
    bits_to_num,
    ModelBuilder,
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
