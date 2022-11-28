from typing import Optional

from housing_data_generator.date_model.example import Features
from housing_model.evaluations.house_price_predictor import HousePricePredictor


class HouseSigmaHousePricePredictor(HousePricePredictor):
    def predict(self, features: Features) -> Optional[float]:
        return features.house_sigma_estimation


class SellerPricePredictor(HousePricePredictor):
    def predict(self, features: Features) -> Optional[float]:
        return features.seller_price
