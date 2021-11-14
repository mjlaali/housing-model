from abc import ABC, abstractmethod
from typing import Optional

from housing_model.data.example import Features, Example


class Model(ABC):
    @abstractmethod
    def predict(self, features: Features) -> Optional[float]:
        pass

    def update(self, example: Example):
        pass
