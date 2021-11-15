from housing_model.models.shallow_model import ModelBuilder, ModelParams
import tensorflow as tf


def test_model_builder_input_output():
    model_builder = ModelBuilder(
        ['f1', 'f2'],
        ModelParams(embedding_size=5)
    )

    model = model_builder.build()
    assert len(model.inputs) == 2
    assert len(model.outputs) == 1


def test_model_builder_generate_price():
    model_builder = ModelBuilder(
        ['f1', 'f2'],
        ModelParams(embedding_size=5)
    )

    model = model_builder.build()
    f1 = tf.constant([1.0])
    f2 = tf.constant([2.0])

    output = model({'f1': f1, 'f2': f2})

    assert output is not None
    assert isinstance(output, dict)
    assert 'sold_price' in output
    assert output['sold_price'].numpy().shape == (1, )


def test_model_overfit():
    model_builder = ModelBuilder(
        ['f1', 'f2'],
        ModelParams(embedding_size=5)
    )

    model = model_builder.build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss='mean_squared_error'
    )

    f1 = tf.constant([1.0, 2.0])
    f2 = tf.constant([2.0, 3.0])
    sold_prices = tf.constant([3.0, 4.0])

    inputs = {'f1': f1, 'f2': f2}

    hist = model.fit(x=inputs, y={'sold_price': sold_prices}, epochs=1000)

    assert hist.history['loss'][-1] < 1e-3

