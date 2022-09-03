import argparse
import json
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds

from housing_model.data.example import Example, Features
from housing_model.evaluations.evaluation import Evaluation, PercentageErrorRate, Metric
from housing_model.models.house_price_predictor import HousePricePredictor
from housing_model.models.keras_model import KerasModelTrainer, TrainParams, ModelParams


def get_overfit_loss(train_ds: tf.data.Dataset, keras_model: KerasModelTrainer, ex_cnt: int) -> float:
    train_ds = train_ds.take(ex_cnt).cache()

    hist = keras_model.fit_model(
        train_ds,
        TrainParams(batch_size=ex_cnt, epochs=500, learning_rate=1e-1),
    )
    return hist.history['loss'][-1]


def eval_model_on_tfds(eval_data: tf.data.Dataset, model: HousePricePredictor) -> Metric:
    examples = [
        Example(ml_num="N/A", sold_price=int(ex["sold_price"].numpy().item()), features=Features(
            house_sigma_estimation=0.0,
            map_lat=ex["map/lat"].numpy().item(),
            map_lon=ex["map/lon"].numpy().item(),
            land_front=ex["land/front"].numpy().item(),
            land_depth=ex["land/depth"].numpy().item(),
            date_end=datetime.fromtimestamp(int(ex["date_end"].numpy().item()) + datetime(1970, 1, 1).timestamp())
        ))
        for ex in eval_data
    ]

    evaluation = Evaluation(PercentageErrorRate, examples)
    metrics = evaluation.eval(model)
    return metrics


def main(model_params_path: str, model_path: str):
    ex_cnt = 4
    train_ds = tfds.load('tf_housing', split='train')

    with open(model_params_path) as fin:
        model_params = ModelParams.from_json(fin.read())

    keras_model = KerasModelTrainer.build(model_params)
    overfit_loss = get_overfit_loss(train_ds, keras_model, ex_cnt)

    keras_model.save(model_path)

    keras_model = KerasModelTrainer.load(model_path)

    predictor = keras_model.make_predictor()
    metric = eval_model_on_tfds(train_ds.take(ex_cnt).cache(), predictor)
    print(json.dumps(metric.value, indent=2, sort_keys=True))

    assert metric.value["mean"] < 0.01, f"The mean percentage error ({metric.value['mean']}) is too high"
    assert overfit_loss < 1e-3, f"The model did not overfit! loss ({overfit_loss}) is too high"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_params_path", required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    main(**vars(args))