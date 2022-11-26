import random
from datetime import datetime
from unittest.mock import MagicMock

import optuna

from housing_model.training.hyper_parameters_tuning import (
    HyperOptSpec,
    HyperParameterObjective,
    VariableSpace,
    hyper_parameters_tuning,
)
from housing_model.training.trainer import ExperimentSpec
from housing_model.training.generators import DateGenerator
from housing_model.modeling.naive_deep.configs import (
    HyperParams,
    ArchitectureParams,
    DatasetSpec,
    DatasetsSpec,
    TrainParams,
    ModelParams,
)


def test_hyper_parameter_tuning():
    config = HyperOptSpec(
        variables=[
            VariableSpace(
                name="lr",
                generator="float_generator",
                generator_config={"start": 0, "end": 1},
            ),
        ],
        name="{lr:.8f}",
        output="dummy_output",
        template=ExperimentSpec(
            datasets=DatasetsSpec(
                DatasetSpec(datetime(2019, 1, 1), datetime(2019, 1, 1)),
                DatasetSpec(datetime(2019, 1, 1), datetime(2019, 1, 1)),
                DatasetSpec(datetime(2019, 1, 1), datetime(2019, 1, 1)),
            ),
            modeling=ModelParams(
                hyper_params=HyperParams(10, 10, 10, 0.1),
                arc_params=ArchitectureParams({"a", "b"}),
            ),
            training=TrainParams(64, 100, learning_rate="{lr}"),
        ).to_dict(),
    )

    def train_op(exp_config: ExperimentSpec, output: str) -> float:
        lr = exp_config.training.learning_rate
        assert 0 <= lr < 1.0
        assert output.endswith(f"{lr:.8f}")
        return lr

    mock_train_op = MagicMock(side_effect=train_op)

    obj = HyperParameterObjective(config, mock_train_op)
    study = optuna.create_study()
    study.optimize(obj, n_trials=10)

    assert mock_train_op.call_count == 10


def test_suggest_date():
    def rnd_gen(name, start, end):
        return start + random.random() * (end - start)

    trial = MagicMock()
    trial.suggest_float = MagicMock(side_effect=rnd_gen)

    generator = DateGenerator("dummy_name", "2010-01-01", "2010-01-02")
    a_date = generator.generate(trial)
    assert a_date == "2010-01-02"


def test_hyper(tmpdir):
    hyper_parameters_tuning(HyperOptSpec([], "name", "output", {}), tmpdir, 10)
