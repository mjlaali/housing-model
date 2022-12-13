import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import dill
import tensorflow as tf
from keras.utils import io_utils

from housing_model.modeling.naive_deep.configs import TrainParams, ModelParams
from housing_model.modeling.naive_deep.data_processor import DatasetProcessor
from housing_model.modeling.naive_deep.model_builder import ModelBuilder
from housing_model.modeling.naive_deep.model_predictor import KerasHousePricePredictor
from housing_model.evaluations.house_price_predictor import HousePricePredictor


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
        additional_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        self.data_provider.setup_preprocessors(train_ds, 10000)

        keras_model = self.data_provider.add_preprocessors(self.keras_model)
        # keras_model = self.keras_model

        loss, loss_callbacks = self._get_loss()
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_params.learning_rate
            ),
            loss=loss,
        )

        callbacks: List[tf.keras.callbacks.Callback] = self._build_callbacks(train_params)
        callbacks += loss_callbacks + additional_callbacks if additional_callbacks else []

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

    def _get_loss(self) -> Tuple[Dict[str, Callable], List[tf.keras.callbacks.Callback]]:
        arc_params = self.model_builder.model_params.arc_params
        bit_loss_weight = self.model_params.hyper_params.bit_loss_weight
        assert 0 <= bit_loss_weight <= 1

        price_loss = AdaptiveLoss(
            "mean_squared_logarithmic_error",
            weight_fn=lambda epoch, w_init: 1 - w_init ** epoch,
            w_init=bit_loss_weight,
            name="w_price",
            verbose=1,
        )
        bit_loss = AdaptiveLoss(
            "binary_crossentropy",
            weight_fn=lambda epoch, w_init: w_init ** epoch,
            w_init=bit_loss_weight,
            name="w_bits",
            verbose=1,
        )
        loss: Dict[str, Callable] = {
            arc_params.price_feature_name: price_loss,
            arc_params.bits_feature_name: bit_loss,
        }
        return loss, [price_loss, bit_loss]

    def _build_callbacks(self, train_params):
        arc_params = self.model_builder.model_params.arc_params
        callbacks = []

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))

        early_stopping_settings = train_params.early_stopping.to_dict()
        if "monitor" not in early_stopping_settings:
            early_stopping_settings[
                "monitor"
            ] = f"val_{arc_params.price_feature_name}_loss"
        callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping_settings))

        return callbacks

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
        keras_model = tf.keras.models.load_model(
            output_dir / "keras_model", custom_objects={"AdaptiveLoss": AdaptiveLoss}
        )

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


class UpdateOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, alpha: tf.Variable, beta: tf.Variable):
        super().__init__()
        self.init_alpha = tf.constant(alpha)
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch, logs=None):
        new_alpha = self.init_alpha ** (epoch // 5)
        self.alpha.assign(new_alpha)
        self.beta.assign(-self.alpha + 1)
        io_utils.print_msg(
            f"\nThe value of alpha and beta are updated to '{self.alpha.numpy()} and {self.beta.numpy()}'."
        )


@dataclass
class AdaptiveLoss(tf.keras.callbacks.Callback):
    loss_fn_id: str
    weight_fn: Callable[[int, float], float]
    w_init: float
    name: str = "weight"
    verbose: int = 0

    def __post_init__(self):
        self._loss_fn = tf.keras.losses.get(self.loss_fn_id)
        self._weight = self.w_init

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self._weight * self._loss_fn(y_true, y_pred)

    def on_epoch_end(self, epoch, logs=None):
        self._weight = self.weight_fn(epoch, self.w_init)
        if self.verbose:
            io_utils.print_msg(f"Weight `{self.name}` got updated to {self._weight}.")

    def get_config(self):
        base64_encode = base64.b64encode(dill.dumps(self.weight_fn))
        return {
            "loss_fn_id": self.loss_fn_id,
            "weight_fn_str": base64_encode.decode("ascii"),
            "w_init": self.w_init,
            "verbose": self.verbose,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        base64_encode = bytes(config.pop("weight_fn_str"), "ascii")
        weight_fn = dill.loads(base64.b64decode(base64_encode))
        config["weight_fn"] = weight_fn
        return cls(**config)
