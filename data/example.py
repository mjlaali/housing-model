import inspect
import json
import typing
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Optional

from dataclasses_json import DataClassJsonMixin


def enforce_types(callable):
    spec = inspect.getfullargspec(callable)

    def check_types(*args, **kwargs):
        parameters = dict(zip(spec.args, args))
        parameters.update(kwargs)
        for name, value in parameters.items():
            with suppress(KeyError):  # Assume un-annotated parameters can be any type
                type_hint = spec.annotations[name]
                if isinstance(type_hint, typing._SpecialForm):
                    # No check for typing.Any, typing.Union, typing.ClassVar (without parameters)
                    continue
                try:
                    actual_type = type_hint.__origin__
                except AttributeError:
                    # In case of non-typing types (such as <class 'int'>, for instance)
                    actual_type = type_hint
                # In Python 3.8 one would replace the try/except with
                # actual_type = typing.get_origin(type_hint) or type_hint
                if isinstance(actual_type, typing._SpecialForm):
                    # case of typing.Union[…] or typing.ClassVar[…]
                    actual_type = type_hint.__args__

                if not isinstance(value, actual_type):
                    raise TypeError(
                        "Unexpected type for '{}' (expected {} but found {})".format(
                            name, type_hint, type(value)
                        )
                    )

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_types(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    if inspect.isclass(callable):
        callable.__init__ = decorate(callable.__init__)
        return callable

    return decorate(callable)


@enforce_types
@dataclass
class Features(DataClassJsonMixin):
    house_sigma_estimation: Optional[float]
    map_lat: float
    map_lon: float
    land_front: Optional[float]
    land_depth: Optional[float]
    date_end: datetime


class InvalidExample(RuntimeError):
    pass


@enforce_types
@dataclass
class Example(DataClassJsonMixin):
    features: Optional[Features]
    sold_price: int

    ml_num: str

    def __post_init__(self):
        if self.sold_price is None or self.ml_num is None:
            raise InvalidExample(json.dumps(self.to_dict()))
