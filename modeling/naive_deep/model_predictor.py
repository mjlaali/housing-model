from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

from housing_data_generator.date_model.example import Features, Example
from housing_model.data.tf_housing import TfHousing
from housing_model.data.tf_housing.feature_names import SOLD_PRICE
from housing_model.evaluations.house_price_predictor import HousePricePredictor


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
