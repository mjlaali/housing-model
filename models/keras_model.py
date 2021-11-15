from dataclasses import dataclass
from typing import Optional, Set

import tensorflow as tf
import numpy as np

from housing_model.data.example import Features, Example
from housing_model.data.tf_housing import TfHousing
from housing_model.models.model import Model


@dataclass
class ModelParams:
    embedding_size: int


@dataclass
class TrainParams:
    batch_size: int
    epochs: int
    learning_rate: float


def bits_to_num(bits, num_bits):
    bits_value = bits * tf.cast(tf.pow(2, tf.range(num_bits)), tf.float32)
    value = tf.reduce_sum(bits_value, axis=-1)
    return value


def num_to_bits(num, num_bits):
  bits = tf.cast(
      tf.bitwise.bitwise_and(
          tf.cast(num, 'int32'),
          tf.pow(2, tf.range(num_bits))
      ) > 0, dtype='int32')
  return bits


@dataclass
class ModelBuilder:
    params: ModelParams
    debug_mode: Optional[bool] = None
    num_bits: int = 32

    def build(self, float_features: Set[str]) -> tf.keras:
        inputs = []
        input_features = []

        for feature in float_features:
            an_input = tf.keras.layers.Input(name=feature, shape=(), dtype='float32')
            inputs.append(an_input)
            expanded_input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(an_input)
            input_embedding = tf.keras.layers.Dense(
                units=self.params.embedding_size,
                activation=tf.math.sin,
                kernel_initializer='random_normal',
                name=f'to_embedding_{feature}'
            )(expanded_input)

            input_feature = tf.keras.layers.Dense(
                units=self.params.embedding_size, activation='relu', name=f'to_feature_l1_{feature}'
            )(input_embedding)

            complete_features = tf.keras.layers.Lambda(
                lambda x: tf.concat(x, axis=-1), name=f"embedding_with_original_{feature}"
            )((input_feature, expanded_input))

            complete_features = tf.keras.layers.Dense(
                units=self.params.embedding_size, activation='relu', name=f'to_feature_l2_{feature}'
            )(complete_features)
            input_features.append(complete_features)

        if len(input_features) > 1:
            features = tf.keras.layers.Add(name="feature_aggregation")(input_features)
        else:
            features = input_features[0]

        sold_price_bits = tf.keras.layers.Dense(units=self.num_bits, activation='sigmoid', name='sold_price')(features)
        sold_price = tf.keras.layers.Lambda(
            lambda bits: bits_to_num(bits, self.num_bits), name='bits'
        )(sold_price_bits)

        # sold_price = tf.keras.layers.Dense(units=1, activation=None, name='dense_price')(features)

        # predictions = tf.keras.layers.Lambda(
        #     lambda batch_prices: tf.squeeze(batch_prices, axis=-1), name='sold_price'
        # )(sold_price)

        self.model = tf.keras.Model(
            inputs=inputs,
            outputs={
                'sold_price': sold_price,
                'bits': sold_price_bits
            }
        )

        self.debug_model = tf.keras.Model(
            inputs=inputs,
            outputs={
                layer.name: layer.output for layer in self.model.layers
            })

        return self.model


class KerasModel(Model):

    @staticmethod
    def build(model_builder: ModelBuilder, train_ds: tf.data.Dataset):
        input_features = set(train_ds.element_spec.keys())
        input_features.remove('metadata')
        input_features.remove('sold_price')
        keras_model = model_builder.build(input_features)

        return KerasModel(
            keras_model,
            train_ds,
            input_features,
            "sold_price"
        )

    def __init__(
            self,
            model: tf.keras.Model,
            train_ds: tf.data.Dataset,
            input_features: Set[str],
            price_feature_name: str,
            num_bits: int = 32
    ):
        self._train_ds = train_ds
        self._model = model
        self._input_features = input_features
        self._price_feature_name = price_feature_name
        self._num_bits = num_bits

    def setup_data(self, tf_data: tf.data.Dataset, batch_size: int):
        return tf_data.map(
            lambda ex: (
                {f_name: ex[f_name] for f_name in self._input_features},
                {
                    self._price_feature_name: ex[self._price_feature_name],
                    'bits': num_to_bits(ex[self._price_feature_name], self._num_bits)
                }
            )
        ).batch(batch_size)

    def train(self, params: TrainParams):
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
            loss={
                #'sold_price': tf.keras.losses.MeanSquaredError(),
                'bits': tf.keras.losses.BinaryCrossentropy()
            }
        )

        self._model.fit(
            self.setup_data(self._train_ds, params.batch_size),
            epochs=params.epochs
        )

    def predict(self, features: Features) -> Optional[float]:
        _, encoded_features = TfHousing.to_features(Example(sold_price=0, ml_num="NA", features=features))
        batch_features = {k: np.asarray([v]) for k, v in encoded_features.items()}
        predictions = self._model.predict(batch_features)
        return predictions[0].astype(float)
