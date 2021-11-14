import argparse
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from glob import glob
from typing import Callable

import numpy as np

from housing_data.analysis.json_to_df import standardize_data
from housing_model.data.data import Data, prepare_data
from housing_model.data.example import Example
from housing_model.models.model import Model
from housing_model.models.baselines import HouseSigmaModel

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
            "cnt": len(values),
            "mean": np.mean(values),
            "med": np.median(values),
            "var": np.std(values),
            "recall": len(values) / self._total_cnt,
            "outlier": self._outlier / self._total_cnt,
        }

    @property
    def values(self):
        return self._values


class Evaluation:
    def __init__(self, metric_factory: Callable[[], Metric], eval_data: Data):
        self._metric_factory = metric_factory
        self._eval_data = eval_data
        self._metric_values = defaultdict(list)

    def eval(self, model: Model):
        metric = self._metric_factory()
        for example in self._eval_data:
            prediction = model.predict(example.features)
            metric.compute(example, prediction)
            model.update(example)

        return metric


def main(eval_file_pattern):
    files = glob(eval_file_pattern)
    cleaned_rows = standardize_data(files)
    examples = prepare_data(cleaned_rows)
    evaluation = Evaluation(PercentageErrorRate, examples)
    model = HouseSigmaModel()

    metrics = evaluation.eval(model)
    print(json.dumps(metrics.value, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_file_pattern")

    args = parser.parse_args()
    main(**vars(args))
