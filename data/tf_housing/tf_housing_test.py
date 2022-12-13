"""tf_housing dataset."""
import logging
import os
import unittest
from datetime import datetime

import tensorflow_datasets as tfds
from . import tf_housing


@unittest.skip("FIXME")
class TfHousingTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for tf_housing dataset."""

    DATASET_CLASS = tf_housing.TfHousing

    SPLITS = {
        "202005": 2,  # Number of fake train example
        "202008": 2,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    DL_EXTRACT_RESULT = {
        2019: ["Y2019/"],
        2020: ["Y2020/"],
    }

    SKIP_CHECKSUMS = True

    def _make_builder(self, config=None):
        return self.DATASET_CLASS(  # pylint: disable=not-callable
            start_time=datetime(2019, 1, 1),
            end_time=datetime(2020, 1, 2),
            data_dir=self.tmp_dir,
            config=config,
            version=self.VERSION,
        )


if __name__ == "__main__":
    os.environ["DATASET_DIR"] = ""
    logging.basicConfig(level=logging.DEBUG)
    tfds.testing.test_main()
