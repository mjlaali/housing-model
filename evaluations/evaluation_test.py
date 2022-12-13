from typing import Dict

import pytest

from housing_data_generator.date_model.example import Example
from housing_model.evaluations.evaluation import PercentageErrorRate


def approx_equal(actual: Dict, expected: Dict):
    assert actual.keys() == expected.keys()

    for k, v in actual.items():
        if isinstance(v, dict):
            approx_equal(actual[k], expected[k])
        else:
            pytest.approx(actual[k], expected[k])

def test_percentage_error():
    metric = PercentageErrorRate()
    for sold, prediction in [(10, 11), (10, 9), (10, None)]:
        metric.compute(
            Example(features=None, ml_num="ml_num", sold_price=sold), prediction
        )

    values = metric.value
    print(values)

    expected = {
        'cnt': 3, 'none_values': {'cnt': 1, 'rate': 0.333}, 'mean': 0.1, 'med': 0.1, 'var': 0.0,
        'hist': {
            'pdf': [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'edges': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'cdf': [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    }
    approx_equal(values, expected)