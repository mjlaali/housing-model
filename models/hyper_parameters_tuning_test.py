from datetime import datetime
from unittest.mock import MagicMock

import optuna

from housing_model.models.hyper_parameters_tuning import (
    HyperOptSpec,
    HyperParameterObjective,
)
from housing_model.models.keras_model import (
    ExperimentSpec,
    DatasetSpec,
    DatasetsSpec,
    ModelParams,
    HyperParams,
    ArchitectureParams,
    TrainParams,
)


def test_hyper_parameter_tuning():
    config = HyperOptSpec(
        variables={
            "lr": "trial.suggest_float('lr', 0, 1)",
            "name": "f\"{trial.suggest_float('lr', 0, 1):.8f}\"",
        },
        name="name",
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
        ),
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
