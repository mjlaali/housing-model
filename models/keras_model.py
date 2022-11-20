from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import numpy as np
import tensorflow as tf
from dataclasses_json import DataClassJsonMixin, config
from keras.utils import io_utils

from housing_data_generator.date_model.example import Features, Example
from housing_model.data.tf_housing import TfHousing
from housing_model.data.tf_housing.feature_names import SOLD_PRICE
from housing_model.models.house_price_predictor import HousePricePredictor


@dataclass
class HyperParams(DataClassJsonMixin):
    embedding_size: int
    num_dense: int
    num_feature_dense: int
    bit_loss_weight: float


@dataclass
class ArchitectureParams(DataClassJsonMixin):
    float_features: Set[str]
    num_bits: int = 32
    price_feature_name: str = "sold_price"
    bits_feature_name: str = "bits"

    @staticmethod
    def from_dataset(train_ds: tf.data.Dataset):
        input_features = set(train_ds.element_spec.keys())
        input_features.remove("metadata")
        input_features.remove("sold_price")
        return ArchitectureParams(input_features)


@dataclass
class EarlyStoppingSetting(DataClassJsonMixin):
    min_delta: float = 0
    patience: int = 100
    verbose: int = 0
    mode: str = "auto"
    restore_best_weights: bool = True


@dataclass
class DatasetSpec(DataClassJsonMixin):
    start: datetime = field(
        metadata=config(
            encoder=lambda date: date.strftime("%Y-%m-%d"),
            decoder=lambda str_date: datetime.strptime(str_date, "%Y-%m-%d"),
        )
    )
    end: datetime = field(
        metadata=config(
            encoder=lambda date: date.strftime("%Y-%m-%d"),
            decoder=lambda str_date: datetime.strptime(str_date, "%Y-%m-%d"),
        )
    )


@dataclass
class DatasetsSpec(DataClassJsonMixin):
    train: DatasetSpec
    dev: DatasetSpec
    test: DatasetSpec


@dataclass
class TrainParams(DataClassJsonMixin):
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping: EarlyStoppingSetting = EarlyStoppingSetting()


@dataclass
class ModelParams(DataClassJsonMixin):
    hyper_params: HyperParams
    arc_params: ArchitectureParams


@dataclass
class ExperimentSpec(DataClassJsonMixin):
    datasets: DatasetsSpec
    training: TrainParams
    modeling: ModelParams


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


@dataclass
class KerasHousePricePredictor(HousePricePredictor):
    """
    Convert a keras model to HousePricePredictor
    """

    model: tf.keras.Model
    price_model_output: str

    def predict(self, features: Features) -> Optional[float]:
        _, encoded_features = TfHousing.to_features(
            Example(sold_price=0, ml_num="NA", features=features)
        )
        batch_features = {
            k: np.asarray([v])
            for k, v in encoded_features.items()
            if k != "metadata" and k != SOLD_PRICE
        }
        predictions = self.model(batch_features, training=False)
        return predictions[self.price_model_output].numpy().astype(float)


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
                    self.model_params.price_feature_name: ex[
                        self.model_params.price_feature_name
                    ],
                    self.model_params.bits_feature_name: num_to_bits(
                        ex[self.model_params.price_feature_name],
                        self.model_params.num_bits,
                    ),
                },
            )
        )

        if batch_size > 0:
            return data.batch(batch_size)
        return data


class UpdateOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, alpha: tf.Variable, beta: tf.Variable):
        super().__init__()
        self.init_alpha = tf.constant(alpha)
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch, logs=None):
        new_alpha = self.init_alpha ** (epoch // 5)
        self.alpha.assign(new_alpha)
        self.beta.assign(1 - self.alpha)
        io_utils.print_msg(
            f"\nThe value of alpha and beta are updated to '{self.alpha.numpy()} and {self.beta.numpy()}'."
        )


@dataclass
class KerasModelTrainer:
    """
    Glue different classes to train a keras model and build a HousePricePredictor
    """

    data_provider: DatasetProcessor
    model_builder: ModelBuilder
    model_params: ModelParams  # This is a redundant and added to ease saving the model configs
    output_dir: Path
    keras_model: Optional[tf.keras.Model] = None

    def __post_init__(self):
        if self.keras_model is None:
            self.keras_model = self.model_builder.build()

    def fit_model(
        self,
        train_ds: tf.data.Dataset,
        dev_ds: tf.data.Dataset,
        train_params: TrainParams,
    ) -> tf.keras.callbacks.History:
        keras_model = self.keras_model

        arc_params = self.model_builder.model_params.arc_params
        bit_loss_weight = self.model_params.hyper_params.bit_loss_weight
        assert 0 <= bit_loss_weight <= 1

        alpha = tf.Variable(bit_loss_weight, name="alpha")
        beta = tf.Variable(1 - alpha, name="beta")
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_params.learning_rate
            ),
            loss={
                arc_params.price_feature_name: tf.keras.losses.MeanSquaredLogarithmicError(),
                arc_params.bits_feature_name: tf.keras.losses.BinaryCrossentropy(),
            },
            loss_weights={
                arc_params.price_feature_name: beta,
                arc_params.bits_feature_name: alpha,
            },
        )

        callbacks = []

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))

        early_stopping_settings = train_params.early_stopping.to_dict()
        if "monitor" not in early_stopping_settings:
            early_stopping_settings[
                "monitor"
            ] = f"val_{arc_params.price_feature_name}_loss"
        callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping_settings))

        callbacks.append(UpdateOnEpoch(alpha, beta))

        # callbacks.append(
        #     tf.keras.callbacks.ModelCheckpoint(
        #         filepath=str(
        #             self.output_dir / "ckpt" / "{epoch:02d}-{val_sold_price_loss:.4f}"
        #         ),
        #         verbose=1,
        #         save_weights_only=True,
        #         monitor="val_sold_price_loss",
        #         mode="min",
        #         save_best_only=True,
        #     )
        # )
        #
        hist = keras_model.fit(
            self.data_provider.setup_data(train_ds, train_params.batch_size),
            validation_data=self.data_provider.setup_data(
                dev_ds, train_params.batch_size
            ),
            epochs=train_params.epochs,
            callbacks=callbacks,
        )
        self.keras_model = keras_model
        return hist

    def make_predictor(self) -> HousePricePredictor:
        return KerasHousePricePredictor(
            self.keras_model,
            self.model_builder.model_params.arc_params.price_feature_name,
        )

    def save(self):
        self.output_dir.mkdir(exist_ok=True)

        with open(self.output_dir / "config.json", "w") as config_file:
            config_file.write(self.model_params.to_json())

        if self.keras_model:
            keras_dir = self.output_dir / "keras_model"
            self.keras_model.save(keras_dir)

    @staticmethod
    def load(output_dir: Path) -> "KerasModelTrainer":
        with open(output_dir / "config.json") as config_file:
            model_params = ModelParams.from_json(config_file.read())

        model_builder = ModelBuilder(model_params)
        keras_model = tf.keras.models.load_model(output_dir / "keras_model")

        return KerasModelTrainer(
            data_provider=DatasetProcessor(model_params.arc_params),
            model_builder=model_builder,
            model_params=model_params,
            output_dir=output_dir,
            keras_model=keras_model,
        )

    @staticmethod
    def build(model_params: ModelParams, output_dir: Path) -> "KerasModelTrainer":
        return KerasModelTrainer(
            DatasetProcessor(model_params.arc_params),
            ModelBuilder(model_params),
            model_params,
            output_dir=output_dir,
        )
