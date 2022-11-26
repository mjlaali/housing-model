import argparse
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from housing_data_generator.date_model.example import Example, Features
from housing_model.data.tf_housing.feature_names import (
    SOLD_PRICE,
    MAP_LAT,
    MAP_LON,
    LAND_FRONT,
    LAND_DEPTH,
    DATE_END,
)
from housing_model.evaluations.evaluation import Metric, Evaluation, PercentageErrorRate
from housing_model.evaluations.house_price_predictor import HousePricePredictor

from housing_model.modeling.naive_deep.model_trainer import KerasModelTrainer


# TODO: write test for this function
def eval_model_on_tfds(
    eval_data: tf.data.Dataset, model: HousePricePredictor
) -> Metric:
    def data_generator():
        for ex in eval_data:
            yield Example(
                ml_num="N/A",
                sold_price=int(ex[SOLD_PRICE].numpy().item()),
                features=Features(
                    house_sigma_estimation=0.0,
                    map_lat=ex[MAP_LAT].numpy().item(),
                    map_lon=ex[MAP_LON].numpy().item(),
                    land_front=ex[LAND_FRONT].numpy().item(),
                    land_depth=ex[LAND_DEPTH].numpy().item(),
                    date_end=datetime.fromtimestamp(
                        int(ex[DATE_END].numpy().item() * 24 * 3600)
                        + datetime(1970, 1, 1).timestamp()
                    ),
                ),
            )

    evaluation = Evaluation(PercentageErrorRate, data_generator())
    metrics = evaluation.eval(model)
    return metrics


def main(model_path: str):
    test_ds = tfds.load("tf_housing", split="test")
    keras_model = KerasModelTrainer.load(Path(model_path))
    predictor = keras_model.make_predictor()
    metrics = eval_model_on_tfds(test_ds, predictor)
    print(json.dumps(metrics.value, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    main(**vars(args))
