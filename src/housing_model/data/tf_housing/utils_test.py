import unittest.mock
from datetime import datetime
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from housing_model.data.tf_housing.utils import path_generator, clean_paths


@pytest.mark.parametrize(
    "start,end,expected",
    [
        pytest.param(
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            [("202001", ["2020/Y2020-sold/days/20200101.jsonl"])],
            id="one_day",
        ),
        pytest.param(
            datetime(2020, 12, 31),
            datetime(2021, 1, 2),
            [
                ("202012", ["2020/Y2020-sold/days/20201231.jsonl"]),
                ("202101", ["2021/Y2021-sold/days/20210101.jsonl"]),
            ],
            id="two_days",
        ),
    ],
)
def test_path_generator(
    start: datetime, end: datetime, expected: List[Tuple[str, List[str]]]
):
    actual = list(path_generator({2020: "2020", 2021: "2021"}, start, end))
    assert actual == expected


@unittest.mock.patch("os.path.exists")
def test_clean_paths(exists_mock: MagicMock):
    exists_mock.side_effect = lambda x: x in {"a.txt"}

    partition_paths = [
        ("list_with_one_valid_file", ["a.txt", "b.txt"]),
        ("no_valid_item", ["b.txt"]),
    ]

    actual = list(clean_paths(partition_paths))

    assert actual == [("list_with_one_valid_file", ["a.txt"])]
