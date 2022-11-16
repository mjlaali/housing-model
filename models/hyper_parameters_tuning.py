import argparse
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Callable, Union

from dataclasses_json import DataClassJsonMixin

from housing_model.models.keras_model import ExperimentSpec

import optuna

import yaml


@dataclass
class HyperOptSpec(DataClassJsonMixin):
    variables: Dict[str, str]
    name: str
    output: str
    template: ExperimentSpec


@dataclass
class HyperParameterObjective:
    config: HyperOptSpec
    train_op: Callable[[ExperimentSpec, str], float]

    def __post_init__(self):
        self._template_str = yaml.dump(self.config.template.to_dict())

    def __call__(self, trial: optuna.Trial) -> float:
        name, exp_config = self._create_exp_config(trial)
        return self.train_op(exp_config, f"{self.config.output}/{name}")

    def _create_exp_config(self, trial: optuna.Trial) -> Tuple[str, ExperimentSpec]:
        variables: Dict[str, Union[int, str]] = {}
        for name, generator in self.config.variables.items():
            val = eval(generator)
            if isinstance(val, str):
                variables[name] = val
            elif isinstance(val, (int, float)):
                variables[name] = val
            else:
                raise ValueError(f"Please convert {type(val)} to compatible json types")

        config_yaml = self._template_str.replace("'{", "{").replace("}'", "}")
        try:
            config_yaml = config_yaml.format(**variables)
            config_dict = yaml.safe_load(config_yaml)
        except KeyError as e:
            raise ValueError(f"cannot format with {str(variables)}") from e

        exp_config = ExperimentSpec.from_dict(config_dict)
        return variables[self.config.name], exp_config


def hyper_parameters_tuning(config: HyperOptSpec):
    study = optuna.create_study()
    study.optimize(HyperParameterObjective(config))


def main(exp_template: str):
    with open(exp_template) as exp_file:
        config = HyperOptSpec.from_json(exp_file.read())

    hyper_parameters_tuning(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-template", required=True)

    args = parser.parse_args()
    main(**vars(args))
