import json
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin


@dataclass
class Features(DataClassJsonMixin):
    house_sigma_estimation: Optional[float]


class InvalidExample(RuntimeError):
    pass


@dataclass
class Example(DataClassJsonMixin):
    features: Optional[Features]
    sold_price: float
    ml_num: str

    def __post_init__(self):
        if self.sold_price is None or self.ml_num is None:
            raise InvalidExample(json.dumps(self.to_dict()))
