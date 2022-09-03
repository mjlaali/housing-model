from typing import Optional

from housing_model.data.example import Features
from housing_model.models.house_price_predictor import HousePricePredictor


class HouseSigmaHousePricePredictor(HousePricePredictor):
    def predict(self, features: Features) -> Optional[float]:
        return features.house_sigma_estimation
