import argparse
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tqdm

from housing_data_model.date_model.data import Data, prepare_data
from housing_data_model.date_model.example import Example
from housing_data_model.date_model.utils import standardize_data, load_from_files
from housing_model.evaluations.house_price_predictor import HousePricePredictor
from housing_model.modeling.baselines import (
    HouseSigmaHousePricePredictor,
    SellerPricePredictor,
)
from housing_model.modeling.naive_deep.model_trainer import KerasModelTrainer

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
        self._none_value = 0

    def compute(self, example: Example, prediction: Optional[float]):
        self._total_cnt += 1
        if prediction is not None:
            val = math.fabs(example.sold_price - prediction) / example.sold_price
            self.values.append(val)
        else:
            self._none_value += 1

    @property
    def value(self) -> dict:
        values = self.values
        bins = np.arange(0, 1, 0.1)
        max_val = np.max(values)
        if max_val > 1.0:
            bins = np.append(bins, max_val)
        pdf, bin_edges = np.histogram(values, bins=bins, density=True)
        cdf = np.cumsum(pdf * np.diff(bin_edges))
        return {
            "cnt": self._total_cnt,
            "none_values": {
                "cnt": self._none_value,
                "rate": self._none_value / self._total_cnt,
            },
            "mean": np.mean(values),
            "med": np.median(values),
            "var": np.std(values),
            "hist": {
                "pdf": pdf.tolist(),
                "edges": bin_edges.tolist(),
                "cdf": cdf.tolist(),
            },
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


baselines = {
    "house_sigma": HouseSigmaHousePricePredictor,
    "seller_price": SellerPricePredictor,
}


def main(eval_file_pattern: str, model_path: Optional[str]):
    paths = glob(eval_file_pattern)
    data_stream = tqdm.tqdm(load_from_files(tqdm.tqdm(paths)))
    cleaned_rows = standardize_data(data_stream)
    examples, _ = prepare_data(cleaned_rows)
    evaluation = Evaluation(PercentageErrorRate, examples)

    if model_path in baselines:
        model = baselines[model_path]()
    else:
        keras_model = KerasModelTrainer.load(Path(model_path))
        model = keras_model.make_predictor()

    metrics = evaluation.eval(model)
    print(json.dumps(metrics.value, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file_pattern", required=True)
    parser.add_argument("--model_path")

    args = parser.parse_args()
    main(**vars(args))
