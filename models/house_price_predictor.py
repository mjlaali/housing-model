from abc import ABC, abstractmethod
from typing import Optional

from housing_model.data.example import Features, Example


class HousePricePredictor(ABC):
    @abstractmethod
    def predict(self, features: Features) -> Optional[float]:
        """
        predict the price given the input features of a single example
        :param features: input features
        :return: the predicted price
        """
        pass

    def update(self, example: Example) -> None:
        """
        Update the predictor with the example
        :param example: the input example
        :return:
        """
        pass
