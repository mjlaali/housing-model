import pytest

from housing_model.data.example import Example, Features
from housing_model.evaluations.evaluation import PercentageErrorRate


def test_percentage_error():
    metric = PercentageErrorRate()
    for sold, prediction in [(10, 11), (10, 9), (10, None)]:
        metric.compute(Example(
            features=None,
            ml_num='ml_num',
            sold_price=sold
        ), prediction)

    values = metric.value
    expected = {
        'cnt': 2,
        'mean': 0.1,
        'med': 0.1,
        'var': 0,
        'recall': 2/3
    }
    for name, val in expected.items():
        pytest.approx(val, values[name])
