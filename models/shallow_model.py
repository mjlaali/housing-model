from dataclasses import dataclass
from typing import Optional, List

import tensorflow_datasets as tfds
import tensorflow as tf
from housing_model.data.example import Features
from housing_model.models.model import Model


@dataclass
class ModelParams:
    embedding_size: int


class ModelBuilder:

    def __init__(self, float_features: List[str], params: ModelParams):
        self._float_features = float_features
        self._params = params

    def build(self) -> tf.keras:
        inputs = []
        input_features = []

        for feature in self._float_features:
            an_input = tf.keras.layers.Input(name=feature, shape=(), dtype='float32')
            inputs.append(an_input)
            expanded_input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(an_input)
            input_embedding = tf.keras.layers.Dense(
                units=self._params.embedding_size,
                activation=tf.math.sin,
                kernel_initializer='random_normal',
                name=f'to_embedding_{feature}'
            )(expanded_input)
            input_embedding = tf.keras.layers.Lambda(
                lambda x: tf.concat(x, axis=-1)
            )((input_embedding, expanded_input))

            input_feature = tf.keras.layers.Dense(
                units=self._params.embedding_size, activation='selu', name=f'to_feature_{feature}'
            )(input_embedding)
            input_features.append(input_feature)

        if len(input_features) > 1:
            features = tf.keras.layers.Add(name="feature_aggregation")(input_features)
        else:
            features = input_features[0]

        sold_price = tf.keras.layers.Dense(units=1, activation='selu', name='dense_price')(features)
        predictions = tf.keras.layers.Lambda(
            lambda batch_prices: tf.squeeze(batch_prices, axis=-1), name='sold_price'
        )(sold_price)

        return tf.keras.Model(
            inputs=inputs,
            outputs={'sold_price': predictions}
        )


class ShallowModel(Model):
    def __init__(self, model_builder: ModelBuilder):
        self.tf_model = model_builder.build()

    def train(self):
        train_ds = tfds.load('tf_housing', split='train')

    def predict(self, features: Features) -> Optional[float]:
        pass
