"""tf_housing dataset."""
from pathlib import Path

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.core.dataset_builder import ReadOnlyPath

from housing_data_generator.utils import standardize_data
from housing_model.data.data import prepare_data

_DESCRIPTION = """
**Housing Data Set**

This data set contains housing features and its sold prices.
"""

_CITATION = """
TBD
"""


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
                {"sold_price": tf.float32, "metadata": {"ml_num": tf.string}}
            ),
            # tfds does not support multiple input features: https://github.com/tensorflow/datasets/issues/849
            supervised_keys=None,  # Set to `None` to disable
            homepage="http://laali.ca",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download(
            "file:///Users/majid/git/housing/housing_data/data/test/"
        )

        return {
            "train": self._generate_examples(path["train"]),
            "test": self._generate_examples(path["test"]),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        files = path.glob("*/data.json")

        cleaned_rows = standardize_data(files)
        examples = prepare_data(cleaned_rows)
        for ex in examples:
            yield ex.ml_num, {
                "sold_price": ex.sold_price,
                "metadata": {"ml_num": ex.ml_num},
            }
