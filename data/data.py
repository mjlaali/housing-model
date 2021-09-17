from typing import Dict, Any, Iterable, Optional

from housing_model.data.example import Example, Features, InvalidExample

Data = Iterable[Example]


def parse(standard_input: Dict[str, Any]) -> Optional[Example]:
    try:
        house_sigma_estimation = standard_input.get('estimate_price')
        return Example(
            features=Features(
                house_sigma_estimation=float(
                    house_sigma_estimation.replace(',', '')) if house_sigma_estimation else None
            ),
            sold_price=standard_input.get('price_sold_int'),
            ml_num=standard_input.get('ml_num'),
        )
    except InvalidExample:
        return None


def prepare_data(standard_inputs: Iterable[Dict]) -> Iterable[Example]:
    examples = map(parse, standard_inputs)
    examples = filter(lambda x: x, examples)
    return examples
