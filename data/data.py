import json
import logging
from datetime import datetime
from typing import Dict, Any, Iterable, Optional, Tuple, Union

from housing_model.data.example import Example, Features, InvalidExample

Data = Iterable[Example]


logger = logging.getLogger(__name__)


class ExampleParser:
    def __init__(self):
        self._err_cnt: int = 0
        self._parsed_examples: int = 0

    @property
    def err_cnt(self) -> int:
        return self._err_cnt

    @property
    def parsed_example(self) -> int:
        return self._parsed_examples

    def parse(self, standard_input: Dict[str, Any]) -> Optional[Example]:
        def to_float(a_num: Optional[Union[int, float]]):
            if a_num is not None:
                return float(a_num)
            return None

        try:
            house_sigma_estimation = standard_input.get("estimate_price")
            example = Example(
                features=Features(
                    house_sigma_estimation=float(
                        house_sigma_estimation.replace(",", "")
                    )
                    if house_sigma_estimation
                    else None,
                    map_lat=standard_input.get("map/lat"),
                    map_lon=standard_input.get("map/lon"),
                    land_front=to_float(standard_input.get("land/front")),
                    land_depth=to_float(standard_input.get("land/depth")),
                    date_end=datetime.strptime(
                        standard_input.get("date_end"), "%Y-%m-%d"
                    ),
                ),
                sold_price=standard_input.get("price_sold_int"),
                ml_num=standard_input.get("ml_num"),
            )
            self._parsed_examples += 1
            return example
        except (InvalidExample, ValueError, TypeError):
            self._err_cnt += 1
            if 9 * self.err_cnt > self.parsed_example:
                logger.exception(
                    f"Cannot parse:\n{json.dumps(standard_input, sort_keys=True, indent=2)}"
                )
            else:
                logger.debug(
                    f"Cannot parse:\n{json.dumps(standard_input, sort_keys=True, indent=2)}",
                    exc_info=True,
                    stack_info=True,
                )

            return None


def prepare_data(
    standard_inputs: Iterable[Dict],
) -> Tuple[Iterable[Example], ExampleParser]:
    parser = ExampleParser()
    examples = map(parser.parse, standard_inputs)
    examples = filter(lambda x: x, examples)
    return examples, parser
