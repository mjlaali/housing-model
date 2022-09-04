from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import numpy as np
import tensorflow as tf
from dataclasses_json import DataClassJsonMixin

from housing_model.data.example import Features, Example
from housing_model.data.tf_housing import TfHousing
from housing_model.models.house_price_predictor import HousePricePredictor


@dataclass
class HyperParams(DataClassJsonMixin):
    embedding_size: int


@dataclass
class ArchitectureParams(DataClassJsonMixin):
    float_features: Set[str]
    num_bits: int = 32
    price_feature_name: str = "sold_price"
    bits_feature_name: str = "bits"

    @staticmethod
    def from_dataset(train_ds: tf.data.Dataset):
        input_features = set(train_ds.element_spec.keys())
        input_features.remove('metadata')
        input_features.remove('sold_price')
        return ArchitectureParams(input_features)


@dataclass
class EarlyStoppingSetting(DataClassJsonMixin):
    min_delta: float = 0
    patience: int = 100
    verbose: int = 0
    mode: str = 'auto'
    restore_best_weights: bool = True


@dataclass
class TrainParams(DataClassJsonMixin):
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping: EarlyStoppingSetting = EarlyStoppingSetting()


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
class ModelParams(DataClassJsonMixin):
    hyper_params: HyperParams
    arc_params: ArchitectureParams


@dataclass
class ModelBuilder(DataClassJsonMixin):
    """
    Build a keras model
    """
    model_params: ModelParams

    debug_mode: Optional[bool] = None

    def build(self) -> tf.keras:
        inputs = []
        input_features = []

        for feature in self.model_params.arc_params.float_features:
            an_input = tf.keras.layers.Input(name=feature, shape=(), dtype='float32')
            inputs.append(an_input)
            expanded_input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(an_input)
            input_embedding = tf.keras.layers.Dense(
                units=self.model_params.hyper_params.embedding_size,
                activation=tf.math.sin,
                kernel_initializer='random_normal',
                name=f'to_embedding_{feature}'
            )(expanded_input)

            input_feature = tf.keras.layers.Dense(
                units=self.model_params.hyper_params.embedding_size, activation='relu', name=f'to_feature_l1_{feature}'
            )(input_embedding)

            complete_features = tf.keras.layers.Lambda(
                lambda x: tf.concat(x, axis=-1), name=f"embedding_with_original_{feature}"
            )((input_feature, expanded_input))

            complete_features = tf.keras.layers.Dense(
                units=self.model_params.hyper_params.embedding_size, activation='relu', name=f'to_feature_l2_{feature}'
            )(complete_features)
            input_features.append(complete_features)

        if len(input_features) > 1:
            features = tf.keras.layers.Add(name="feature_aggregation")(input_features)
        else:
            features = input_features[0]

        sold_price_bits = tf.keras.layers.Dense(
            units=self.model_params.arc_params.num_bits, activation='sigmoid',
            name=self.model_params.arc_params.bits_feature_name)(features)
        sold_price = tf.keras.layers.Lambda(
            lambda bits: bits_to_num(bits, self.model_params.arc_params.num_bits),
            name=self.model_params.arc_params.price_feature_name
        )(sold_price_bits)

        # sold_price = tf.keras.layers.Dense(units=1, activation=None, name='dense_price')(features)

        # predictions = tf.keras.layers.Lambda(
        #     lambda batch_prices: tf.squeeze(batch_prices, axis=-1), name='sold_price'
        # )(sold_price)

        self.model = tf.keras.Model(
            inputs=inputs,
            outputs={
                self.model_params.arc_params.price_feature_name: sold_price,
                self.model_params.arc_params.bits_feature_name: sold_price_bits
            }
        )

        self.debug_model = tf.keras.Model(
            inputs=inputs,
            outputs={
                layer.name: layer.output for layer in self.model.layers
            })

        return self.model


@dataclass
class KerasHousePricePredictor(HousePricePredictor):
    """
    Convert a keras model to HousePricePredictor
    """
    model: tf.keras.Model
    price_model_output: str

    def predict(self, features: Features) -> Optional[float]:
        _, encoded_features = TfHousing.to_features(Example(sold_price=0, ml_num="NA", features=features))
        batch_features = {k: np.asarray([v]) for k, v in encoded_features.items() if k != "metadata"}
        predictions = self.model.predict(batch_features)
        return predictions[self.price_model_output].astype(float)


@dataclass
class DatasetProcessor:
    """
    Prepare a data set for a specific model.
    """
    model_params: ArchitectureParams

    def setup_data(self, tf_data: tf.data.Dataset, batch_size: int):
        data = tf_data.map(
            lambda ex: (
                {f_name: ex[f_name] for f_name in self.model_params.float_features},
                {
                    self.model_params.price_feature_name: ex[self.model_params.price_feature_name],
                    self.model_params.bits_feature_name: num_to_bits(
                        ex[self.model_params.price_feature_name], self.model_params.num_bits
                    )
                }
            )
        )

        if batch_size > 0:
            return data.batch(batch_size)
        return data


@dataclass
class KerasModelTrainer:
    """
    Glue different classes to train a keras model and build a HousePricePredictor
    """

    data_provider: DatasetProcessor
    model_builder: ModelBuilder
    model_params: ModelParams  # This is a redundant and added to ease saving the model configs
    keras_model: Optional[tf.keras.Model] = None

    def fit_model(self,
                  train_ds: tf.data.Dataset,
                  dev_ds: tf.data.Dataset,
                  train_params: TrainParams) -> tf.keras.callbacks.History:
        keras_model = self.model_builder.build()
        arc_params = self.model_builder.model_params.arc_params
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=train_params.learning_rate),
            loss={
                arc_params.price_feature_name: tf.keras.losses.MeanSquaredLogarithmicError(),
                arc_params.bits_feature_name: tf.keras.losses.BinaryCrossentropy()
            },
            loss_weights={
                arc_params.price_feature_name: 0.99,
                arc_params.bits_feature_name: 0.01
            }
        )

        callbacks = []
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))

        early_stopping_settings = train_params.early_stopping.to_dict()
        early_stopping_settings["monitor"] = f"{arc_params.price_feature_name}_loss"
        callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping_settings))

        hist = keras_model.fit(
            self.data_provider.setup_data(train_ds, train_params.batch_size),
            epochs=train_params.epochs,
            callbacks=callbacks
        )
        self.keras_model = keras_model
        return hist

    def make_predictor(self) -> HousePricePredictor:
        return KerasHousePricePredictor(
            self.keras_model, self.model_builder.model_params.arc_params.price_feature_name
        )

    def save(self, a_path: str):
        path_dir = Path(a_path)
        path_dir.mkdir(exist_ok=True)

        with open(path_dir / "config.json", "w") as config_file:
            config_file.write(self.model_params.to_json())

        if self.keras_model:
            keras_dir = path_dir / "keras_model"
            keras_dir.mkdir(exist_ok=True)
            self.keras_model.save_weights(keras_dir)

    @staticmethod
    def load(a_path: str) -> "KerasModelTrainer":
        path_dir = Path(a_path)
        with open(path_dir / "config.json") as config_file:
            model_params = ModelParams.from_json(config_file.read())

        model_builder = ModelBuilder(model_params)
        keras_model = model_builder.build()
        keras_model.load_weights(path_dir / "keras_model")

        return KerasModelTrainer(
            DatasetProcessor(model_params.arc_params),
            model_builder,
            model_params,
            keras_model
        )

    @staticmethod
    def build(model_params: ModelParams) -> "KerasModelTrainer":
        return KerasModelTrainer(
            DatasetProcessor(model_params.arc_params),
            ModelBuilder(model_params),
            model_params
        )

