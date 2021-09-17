from abc import ABC, abstractmethod
from typing import Optional

from housing_model.data.example import Features, Example


class Model(ABC):
    @abstractmethod
    def predict(self, features: Features) -> Optional[float]:
        pass

    def update(self, example: Example):
        pass


class HouseSigmaModel(Model):
    def predict(self, features: Features) -> Optional[float]:
        return features.house_sigma_estimation
