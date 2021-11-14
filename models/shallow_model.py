from typing import Optional

from housing_model.data.data import Data
from housing_model.data.example import Features
from housing_model.models.model import Model


class ShallowModel(Model):
    def __init__(self):

        pass

    def train(self, train_data: Data):
        pass

    def predict(self, features: Features) -> Optional[float]:
        pass
