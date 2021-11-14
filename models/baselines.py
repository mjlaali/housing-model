from typing import Optional

from housing_model.data.example import Features
from housing_model.models.model import Model


class HouseSigmaModel(Model):
    def predict(self, features: Features) -> Optional[float]:
        return features.house_sigma_estimation
