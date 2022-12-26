from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from optuna import Trial


@dataclass
class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, trial: Trial) -> Any:
        pass


@dataclass
class FloatGenerator(Generator):
    start: float
    end: float

    def generate(self, trial: Trial) -> Any:
        return trial.suggest_float(self.name, self.start, self.end)


@dataclass
class IntGenerator(Generator):
    start: int
    end: int

    def generate(self, trial: Trial) -> Any:
        return trial.suggest_int(self.name, self.start, self.end)


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
        random_day = trial.suggest_float(self.name, 0, self._day_range)
        total_secs = (self._end_days - random_day) * 3600 * 24
        date_str = datetime.fromtimestamp(total_secs).strftime("%Y-%m-%d")
        return date_str
