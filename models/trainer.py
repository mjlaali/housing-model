import argparse
import json
import shutil
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds

from housing_model.data.example import Example, Features
from housing_model.data.tf_housing.feature_names import SOLD_PRICE, MAP_LAT, MAP_LON, LAND_FRONT, LAND_DEPTH, DATE_END
from housing_model.evaluations.evaluation import Evaluation, PercentageErrorRate, Metric
from housing_model.models.house_price_predictor import HousePricePredictor
from housing_model.models.keras_model import KerasModelTrainer, TrainParams, ModelParams, EarlyStoppingSetting


def get_overfit_loss(train_ds: tf.data.Dataset, keras_model: KerasModelTrainer, overfit_train_params: TrainParams) -> float:
    dataset_size = len(list(train_ds))
    hist = keras_model.fit_model(
        train_ds,
        train_ds.take(dataset_size).cache(),
        overfit_train_params,
    )
    return hist.history['loss'][-1]


# TODO: write test for this function
def eval_model_on_tfds(eval_data: tf.data.Dataset, model: HousePricePredictor) -> Metric:
    examples = [
        Example(ml_num="N/A", sold_price=int(ex[SOLD_PRICE].numpy().item()), features=Features(
            house_sigma_estimation=0.0,
            map_lat=ex[MAP_LAT].numpy().item(),
            map_lon=ex[MAP_LON].numpy().item(),
            land_front=ex[LAND_FRONT].numpy().item(),
            land_depth=ex[LAND_DEPTH].numpy().item(),
            date_end=datetime.fromtimestamp(
                int(ex[DATE_END].numpy().item() * 24 * 3600) + datetime(1970, 1, 1).timestamp()
            )
        ))
        for ex in eval_data
    ]

    evaluation = Evaluation(PercentageErrorRate, examples)
    metrics = evaluation.eval(model)
    return metrics


def main(model_params_path: str, model_path: str, train_params_path: str, overfit_train_params_path: str):
    train_ds = tfds.load('tf_housing', split='train')

    with open(model_params_path) as fin:
        model_params = ModelParams.from_json(fin.read())

    with open(overfit_train_params_path) as fin:
        overfit_train_params = TrainParams.from_json(fin.read())

    # check the model architecture does not have any error
    check_model_architecture(model_params, model_path, train_ds, overfit_train_params)
    shutil.rmtree(model_path)

    # start training job and export the model
    dev_ds = tfds.load('tf_housing', split='dev')
    keras_model = KerasModelTrainer.build(model_params)
    with open(train_params_path) as fin:
        train_params = TrainParams.from_json(fin.read())
    keras_model.fit_model(train_ds, dev_ds, train_params)

    # test the exported model
    test_ds = tfds.load('tf_housing', split='test')
    keras_model = KerasModelTrainer.load(model_path)
    predictor = keras_model.make_predictor()
    metrics = eval_model_on_tfds(test_ds, predictor)
    print(json.dumps(metrics.value, indent=2, sort_keys=True))


def check_model_architecture(
        model_params: ModelParams, model_path: str, train_ds: tf.data.Dataset, overfit_train_params: TrainParams
):
    ex_cnt = overfit_train_params.batch_size
    train_ds = train_ds.take(ex_cnt).cache()
    test_ds = train_ds.take(ex_cnt).cache()

    keras_model = KerasModelTrainer.build(model_params)

    overfit_loss = get_overfit_loss(train_ds, keras_model, overfit_train_params)

    keras_model.save(model_path)
    # keras_model = KerasModelTrainer.load(model_path)
    predictor = keras_model.make_predictor()

    metric = eval_model_on_tfds(test_ds, predictor)
    print(json.dumps(metric.value, indent=2, sort_keys=True))
    assert metric.value["mean"] < 0.01, f"The mean percentage error ({metric.value['mean']}) is too high"
    assert overfit_loss < 1e-3, f"The model did not overfit! loss ({overfit_loss}) is too high"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_params_path", required=True)
    parser.add_argument("--train_params_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--overfit_train_params_path", required=True)

    args = parser.parse_args()

    main(**vars(args))