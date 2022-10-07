import argparse
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from glob import glob
from typing import Callable, Optional

import numpy as np
import tqdm

from housing_data_generator.date_model.data import Data, prepare_data
from housing_data_generator.date_model.example import Example
from housing_data_generator.date_model.utils import standardize_data
from housing_model.models.house_price_predictor import HousePricePredictor
from housing_model.models.baselines import HouseSigmaHousePricePredictor
from housing_model.models.keras_model import KerasModelTrainer

_logger = logging.getLogger(__name__)


class Metric(ABC):
    @abstractmethod
    def compute(self, example: Example, prediction: float):
        pass

    @property
    @abstractmethod
    def value(self) -> dict:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PercentageErrorRate(Metric):
    def __init__(self):
        self._values = []
        self._total_cnt = 0
        self._outlier = 0

    def compute(self, example: Example, prediction: float):
        self._total_cnt += 1
        if prediction is not None:
            val = math.fabs(example.sold_price - prediction) / example.sold_price
            if val < 1.0:
                self.values.append(val)
            else:
                self._outlier += 1

    @property
    def value(self) -> dict:
        values = self.values
        return {
            "cnt": self._total_cnt,
            "mean": np.mean(values),
            "med": np.median(values),
            "var": np.std(values),
            "outlier_cnt": self._outlier,
            "in_range_cnt": len(values),
            "in_range_rate": len(values) / self._total_cnt,
            "outlier_rate": self._outlier / self._total_cnt,
        }

    @property
    def values(self):
        return self._values


class Evaluation:
    def __init__(self, metric_factory: Callable[[], Metric], eval_data: Data):
        self._metric_factory = metric_factory
        self._eval_data = eval_data
        self._metric_values = defaultdict(list)

    def eval(self, model: HousePricePredictor) -> Metric:
        metric = self._metric_factory()

        for example in tqdm.tqdm(self._eval_data):
            prediction = model.predict(example.features)
            metric.compute(example, prediction)
            model.update(example)

        return metric


def main(eval_file_pattern: str, model_path: Optional[str]):
    files = glob(eval_file_pattern)
    cleaned_rows = standardize_data(files)
    examples, _ = prepare_data(cleaned_rows)
    evaluation = Evaluation(PercentageErrorRate, examples)

    if model_path:
        keras_model = KerasModelTrainer.load(model_path)
        model = keras_model.make_predictor()
    else:
        model = HouseSigmaHousePricePredictor()

    metrics = evaluation.eval(model)
    print(json.dumps(metrics.value, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file_pattern", required=True)
    parser.add_argument("--model_path")

    args = parser.parse_args()
    main(**vars(args))
