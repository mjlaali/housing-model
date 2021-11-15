from collections import OrderedDict
from pprint import pprint

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from housing_model.models.keras_model import ModelBuilder, ModelParams, KerasModel, TrainParams, bits_to_num


def test_bits_to_num():
    bits = tf.constant([
        [1, 0, 1],
        [0, 1, 0]
    ], dtype='float32')
    res = bits_to_num(bits, 3)
    np.testing.assert_almost_equal(res.numpy(), [5, 2])


def test_model_builder_input_output():
    model_builder = ModelBuilder(
        ModelParams(embedding_size=5)
    )

    model = model_builder.build({'f1', 'f2'})
    assert len(model.inputs) == 2
    assert len(model.outputs) == 2


def test_model_builder_generate_price():
    model_builder = ModelBuilder(
        ModelParams(embedding_size=5)
    )

    model = model_builder.build({'f1', 'f2'})
    f1 = tf.constant([1.0])
    f2 = tf.constant([2.0])

    output = model({'f1': f1, 'f2': f2})

    assert output is not None
    assert isinstance(output, dict)
    assert 'sold_price' in output
    assert output['sold_price'].numpy().shape == (1,)

    assert 'bits' in output
    assert output['bits'].numpy().shape == (1, 32)


def test_model_overfit():
    num_bits = 3
    model_builder = ModelBuilder(
        ModelParams(embedding_size=5), num_bits=num_bits
    )

    model = model_builder.build({'f1', 'f2'})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-2,
            clipnorm=1.0
        ),
        loss={
            #'sold_price': 'mean_squared_error',
            'bits': 'binary_crossentropy'
        }
    )

    f1 = tf.constant([1.0, 2.0])
    f2 = tf.constant([2.0, 3.0])
    sold_prices = tf.constant([3.0, 4.0])
    bits = tf.constant([[0, 1, 1], [1, 0, 0]])

    inputs = {'f1': f1, 'f2': f2}

    hist = model.fit(x=inputs, y={'sold_price': sold_prices, 'bits': bits}, epochs=1000)

    assert hist.history['loss'][-1] < 1e-3


def test_train():
    train_ds = tfds.load('tf_housing', split='train').take(6).cache()

    model_builder = ModelBuilder(ModelParams(embedding_size=20))
    keras_model = KerasModel.build(model_builder, train_ds)
    test_ds = keras_model.setup_data(train_ds.take(1), batch_size=1)

    keras_model.train(TrainParams(batch_size=3, epochs=5000, learning_rate=1e-1))
    output = keras_model._model.predict(test_ds.map(lambda x, y: x))
    for idx, (_, expected) in enumerate(test_ds):
        for k, v in expected.items():
            print(f'{k}:\n{v}\n')
        print(f'expected: {expected["sold_price"]}\nres:\n{output[1][idx]}')
        for expected_bits, res_bit in zip(expected["bits"][0], output[0][idx]):
            print(f'{expected_bits} - {res_bit}')
        # print(f'bits:\n{output[0][idx]}')

    #pprint(output)
    assert False

    #keras_model.train(TrainParams(batch_size=3, epochs=1000, learning_rate=1e-1))
    #run_debug_job(keras_model, model_builder, test_ds)


def run_debug_job(keras_model, model_builder, test_ds):
    # print(f'dense layer weights:\n{str(model_builder.model.get_layer("dense_price").weights)}')

    predictions = None
    for a_batch in test_ds:
        predictions = model_builder.debug_model(a_batch)
    predictions = OrderedDict({layer.name: predictions[layer.name] for layer in keras_model._model.layers})
    print('layer outputs:\n')
    pprint(predictions)
