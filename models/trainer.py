import argparse
import json
import shutil

import tensorflow as tf
import tensorflow_datasets as tfds

from housing_model.evaluations.keras_model_evaluator import eval_model_on_tfds
from housing_model.models.keras_model import KerasModelTrainer, TrainParams, ModelParams


def get_overfit_loss(train_ds: tf.data.Dataset, keras_model: KerasModelTrainer, overfit_train_params: TrainParams) -> float:
    dataset_size = len(list(train_ds))
    hist = keras_model.fit_model(
        train_ds,
        train_ds.take(dataset_size).cache(),
        overfit_train_params,
    )
    return hist.history['loss'][-1]


def main(model_params_path: str, model_path: str, train_params_path: str, overfit_train_params_path: str):
    train_ds = tfds.load('tf_housing', split='train')

    with open(model_params_path) as fin:
        model_params = ModelParams.from_json(fin.read())

    with open(overfit_train_params_path) as fin:
        overfit_train_params = TrainParams.from_json(fin.read())

    # check the model architecture does not have any error
    check_model_architecture(model_params, model_path, train_ds, overfit_train_params)

    # start training job and export the model
    dev_ds = tfds.load('tf_housing', split='dev')
    keras_model = KerasModelTrainer.build(model_params)
    with open(train_params_path) as fin:
        train_params = TrainParams.from_json(fin.read())
    keras_model.fit_model(train_ds, dev_ds, train_params)

    shutil.rmtree(model_path)
    keras_model.save(model_path)

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