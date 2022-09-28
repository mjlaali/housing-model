"""tf_housing dataset."""
import logging
from datetime import datetime
import os
from typing import Tuple, Dict, Any

import tensorflow as tf
import tensorflow_datasets as tfds

from housing_data_generator.utils import standardize_data
from housing_model.data.data import prepare_data
from housing_model.data.example import Example
from housing_model.data.tf_housing.feature_names import (
    SOLD_PRICE,
    MAP_LAT,
    MAP_LON,
    LAND_FRONT,
    LAND_DEPTH,
    DATE_END,
    METADATA,
    ML_NUM,
)

_DESCRIPTION = """
**Housing Data Set**

This data set contains housing features and its sold prices.
"""

_CITATION = """
TBD
"""

logger = logging.getLogger(__name__)


class TfHousing(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for tf_housing dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    SOLD_PRICE: tf.float32,
                    MAP_LAT: tf.float32,
                    MAP_LON: tf.float32,
                    LAND_FRONT: tf.float32,
                    LAND_DEPTH: tf.float32,
                    DATE_END: tf.float32,
                    METADATA: {ML_NUM: tf.string},
                }
            ),
            # tfds does not support multiple input features: https://github.com/tensorflow/datasets/issues/849
            supervised_keys=None,  # Set to `None` to disable
            homepage="http://laali.ca",
            citation=_CITATION,
        )

    def _get_housing_data_dir(self):
        # window = file:///Users/majid/git/housing/
        housing_dir = os.getenv(
            "DATASET_DIR", f"{os.path.dirname(__file__)}/../../../housing_data"
        )
        return os.path.abspath(housing_dir)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        paths = dl_manager.download_and_extract(
            {
                "train": [f"{self._get_housing_data_dir()}/Y2019-sold.tar.gz"],
                "dev": [f"{self._get_housing_data_dir()}/Y2018-sold.tar.gz"],
                "test": [f"{self._get_housing_data_dir()}/Y2020-sold.tar.gz"],
            }
        )

        return {
            split: self._generate_examples(paths[split])
            for split in ("train", "test", "dev")
        }

    @staticmethod
    def to_features(ex: Example) -> Tuple[str, Dict[str, Any]]:
        """
        parse an example and generate a list of features
        :param ex: The input example
        :return: A tuple of example_id (i.e. MLNum) and a dict of its features
        """
        return ex.ml_num, {
            SOLD_PRICE: ex.sold_price,
            MAP_LAT: ex.features.map_lat,
            MAP_LON: ex.features.map_lon,
            LAND_FRONT: ex.features.land_front or 1,  # Convert missing values to 1
            LAND_DEPTH: ex.features.land_depth or 1,  # Convert missing values to 1
            DATE_END: (ex.features.date_end - datetime(1970, 1, 1)).total_seconds()
            // 3600
            // 24,
            METADATA: {ML_NUM: ex.ml_num},
        }

    def _generate_examples(self, paths):
        """Yields examples."""
        for path in paths:
            files = list(path.glob("*/data.json"))

            cleaned_rows = standardize_data(files)
            examples, parser = prepare_data(cleaned_rows)
            for ex in examples:
                yield self.to_features(ex)
            logger.info(
                f"{parser.parsed_example} has been read from {str(files)}. "
                f"{parser.err_cnt} examples have been filtered due to a parse error. "
                f"{parser.err_cnt / max(parser.parsed_example + parser.err_cnt, 1):.2f}"
            )
