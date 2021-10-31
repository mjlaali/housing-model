"""tf_housing dataset."""

import tensorflow_datasets as tfds
from . import tf_housing


class TfHousingTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for tf_housing dataset."""

    DATASET_CLASS = tf_housing.TfHousing
    SPLITS = {
        "train": 2,  # Number of fake train example
        "test": 1,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    DL_EXTRACT_RESULT = {
        "train": "train/",
        "test": "test/",
    }

    SKIP_CHECKSUMS = True


if __name__ == "__main__":
    tfds.testing.test_main()
