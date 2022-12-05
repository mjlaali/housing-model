from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf

from housing_model.modeling.naive_deep.configs import ModelParams


def bits_to_num(bits, num_bits):
    bits_value = bits * tf.cast(tf.pow(2, tf.range(num_bits)), tf.float32)
    value = tf.reduce_sum(bits_value, axis=-1)
    return value


def num_to_bits(num, num_bits):
    bits = tf.cast(
        tf.bitwise.bitwise_and(tf.cast(num, "int32"), tf.pow(2, tf.range(num_bits)))
        > 0,
        dtype="int32",
    )
    return bits


@dataclass
class ModelBuilder:
    """
    Build a keras model
    """

    model_params: ModelParams
    model: Optional[tf.keras.Model] = field(init=False)

    def build(self) -> tf.keras:
        inputs = []
        input_features = []

        for feature in self.model_params.arc_params.float_features:
            an_input = tf.keras.layers.Input(name=feature, shape=(), dtype="float32")
            inputs.append(an_input)
            expanded_input = tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(x, axis=-1)
            )(an_input)
            input_embedding = tf.keras.layers.Dense(
                units=self.model_params.hyper_params.embedding_size,
                activation=tf.math.sin,
                kernel_initializer="random_normal",
                name=f"to_embedding_{feature}",
            )(expanded_input)

            input_feature = tf.keras.layers.Dense(
                units=self.model_params.hyper_params.embedding_size,
                activation="relu",
                name=f"to_feature_{feature}",
            )(input_embedding)

            for i in range(self.model_params.hyper_params.num_feature_dense):
                input_feature += tf.keras.layers.Dense(
                    units=self.model_params.hyper_params.embedding_size,
                    activation="relu",
                    name=f"to_feature_l{i}_{feature}",
                )(input_feature)

            input_features.append(input_feature)

        if len(input_features) > 1:
            features = tf.keras.layers.Add(name="feature_aggregation")(input_features)
        else:
            features = input_features[0]

        dense_features = tf.keras.layers.Dense(
            units=self.model_params.arc_params.num_bits,
            activation="relu",
            name="dense_features",
        )(features)
        for i in range(self.model_params.hyper_params.num_dense):
            dense_features = (
                tf.keras.layers.Dense(
                    units=self.model_params.arc_params.num_bits,
                    activation="leaky_relu",
                    name=f"Dens-{i}",
                )(dense_features)
                + dense_features
            )

        sold_price_bits = tf.keras.layers.Dense(
            units=self.model_params.arc_params.num_bits,
            activation="sigmoid",
            name=self.model_params.arc_params.bits_feature_name,
        )(features)
        num_bits = self.model_params.arc_params.num_bits
        sold_price = tf.keras.layers.Lambda(
            lambda bits: bits_to_num(bits, num_bits),
            name=self.model_params.arc_params.price_feature_name,
        )(sold_price_bits)

        self.model = tf.keras.Model(
            inputs=inputs,
            outputs={
                self.model_params.arc_params.price_feature_name: sold_price,
                self.model_params.arc_params.bits_feature_name: sold_price_bits,
            },
        )

        return self.model

    def adapt(self, dataset: tf.data.Dataset):
        pass
