import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Callable, Union, List

import numpy as np
from datetime import datetime

from dataclasses_json import DataClassJsonMixin, config
from optuna import Trial

from housing_model.models.keras_model import ExperimentSpec

import optuna

import yaml

from housing_model.models.trainer import train_job


@dataclass
class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, trial: Trial) -> Any:
        pass


@dataclass
class FloatGenerator(Generator):
    start: int
    end: int

    def generate(self, trial: Trial) -> Any:
        return trial.suggest_float(self.name, self.start, self.end)


@dataclass
class DateGenerator(Generator):
    start: str
    end: str

    def __post_init__(self):
        start_date = datetime.fromisoformat(self.start)
        end_date = datetime.fromisoformat(self.end)
        base = datetime.fromtimestamp(0)
        self._end_days = (end_date - base).total_seconds() / 3600 / 24
        start_days = (start_date - base).total_seconds() / 3600 / 24
        self._day_range = self._end_days - start_days

    def generate(self, trial: Trial) -> Any:
        random_day = trial.suggest_int(self.name, 0, self._day_range)
        total_secs = (self._end_days - random_day) * 3600 * 24
        date_str = datetime.fromtimestamp(total_secs).strftime("%Y-%m-%d")
        return date_str


suggester_factory = {"date_generator": DateGenerator, "float_generator": FloatGenerator}


@dataclass
class VariableSpace(DataClassJsonMixin):
    name: str
    generator: str
    generator_config: Dict[str, Any]

    def make_generator(self):
        suggester_cls = suggester_factory[self.generator]
        return suggester_cls(self.name, **self.generator_config)


@dataclass
class HyperOptSpec(DataClassJsonMixin):
    variables: List[VariableSpace]
    name: str
    output: str
    template: Dict[str, Any]


@dataclass
class HyperParameterObjective:
    config: HyperOptSpec
    train_op: Callable[[ExperimentSpec, str], float]

    def __post_init__(self):
        self._template_str = (
            yaml.dump(self.config.template).replace("'{", "{").replace("}'", "}")
        )
        self._variable_generator = {
            var.name: var.make_generator() for var in self.config.variables
        }

    def __call__(self, trial: optuna.Trial) -> float:
        name, exp_config = self._create_exp_config(trial)
        return self.train_op(exp_config, f"{self.config.output}/{name}")

    def _create_exp_config(self, trial: optuna.Trial) -> Tuple[str, ExperimentSpec]:
        variables: Dict[str, Union[int, str]] = {}
        for name, generator in self._variable_generator.items():
            val = generator.generate(trial)
            if isinstance(val, str):
                variables[name] = f"'{val}'"
            elif isinstance(val, (int, float)):
                variables[name] = val
            else:
                raise ValueError(f"Please convert {type(val)} to compatible json types")

        config_yaml = self._template_str
        try:
            config_yaml = config_yaml.format(**variables)
            config_dict = yaml.safe_load(config_yaml)
        except KeyError as e:
            raise ValueError(f"cannot format with {str(variables)}") from e

        exp_config = ExperimentSpec.from_dict(config_dict)
        trial_name = self.config.name.format(**variables)
        return trial_name, exp_config


def hyper_parameters_tuning(config: HyperOptSpec, n_trials: int):
    study_name = config.output  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True
    )

    study.optimize(HyperParameterObjective(config, train_job), n_trials=n_trials)


def main(exp_template: str, n_trials: int):
    with open(exp_template) as exp_file:
        config = HyperOptSpec.from_json(exp_file.read())

    hyper_parameters_tuning(config, n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-template", required=True)
    parser.add_argument("--n-trials", type=int, required=True)

    args = parser.parse_args()
    main(**vars(args))
