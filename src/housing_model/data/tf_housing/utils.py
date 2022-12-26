import datetime
import os
from typing import Dict, List, Tuple, Iterable

import logging

logger = logging.getLogger(__name__)


def path_generator(
    base_paths: Dict[int, str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> Iterable[Tuple[str, List[str]]]:
    assert start_date < end_date
    a_date = start_date
    current_month = a_date.month

    while a_date < end_date:
        file_paths: List[str] = []
        partition = a_date.strftime("%Y%m")

        while a_date.month == current_month and a_date < end_date:
            file_paths.append(
                f"{base_paths[a_date.year]}/Y{a_date.year}-sold/days/{a_date.strftime('%Y%m%d')}.jsonl"
            )
            a_date += datetime.timedelta(days=1)
        current_month = a_date.month

        yield partition, file_paths


def clean_paths(partition_paths: Iterable[Tuple[str, List[str]]]):
    partition_paths = map(
        lambda partition_files: (
            partition_files[0],
            [path for path in partition_files[1] if os.path.exists(path)],
        ),
        partition_paths,
    )

    partition_paths = filter(
        lambda partition_files: partition_files[1], partition_paths
    )

    partition_paths = list(partition_paths)
    return partition_paths
