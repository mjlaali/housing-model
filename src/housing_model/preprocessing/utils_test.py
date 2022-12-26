from housing_model.preprocessing.utils import to_example
import tensorflow as tf
import numpy as np


def test_to_example():
    features = {
        "int": np.asarray([1, 2, 3], dtype=np.int32),
        "float": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "string": ["a", "b", "c"],
    }

    example = to_example(features)
    assert isinstance(example, tf.train.Example)
