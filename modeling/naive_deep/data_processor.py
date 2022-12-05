import logging
from dataclasses import dataclass, field
from typing import Dict

import tensorflow as tf

from housing_model.modeling.naive_deep.configs import ArchitectureParams
from housing_model.modeling.naive_deep.model_builder import num_to_bits

_logger = logging.getLogger(__name__)


@dataclass
class DatasetProcessor:
    """
    Prepare a data set for a specific model.
    """

    model_params: ArchitectureParams
    preprocessors: Dict[
        str, tf.keras.layers.experimental.preprocessing.PreprocessingLayer
    ] = field(init=False, default_factory=lambda: {})

    def setup_data(self, tf_data: tf.data.Dataset, batch_size: int):
        self.setup_preprocessors(tf_data)
        preprocessors = self.preprocessors

        data = tf_data.map(
            lambda ex: (
                {
                    f_name: preprocessors[f_name](ex[f_name])
                    for f_name in self.model_params.float_features
                },
                {
                    self.model_params.price_feature_name: ex[
                        self.model_params.price_feature_name
                    ],
                    self.model_params.bits_feature_name: num_to_bits(
                        ex[self.model_params.price_feature_name],
                        self.model_params.num_bits,
                    ),
                },
            )
        )

        if batch_size > 0:
            return data.batch(batch_size)
        return data

    def setup_preprocessors(self, tf_data: tf.data.Dataset):
        _logger.info("Initializing preprocessors ...")
        cached_data = tf_data.take(10000)
        for f_name in self.model_params.float_features:
            self.preprocessors[f_name] = tf.keras.layers.Normalization(axis=None)
            self.preprocessors[f_name].adapt(cached_data.map(lambda ex: ex[f_name]))
        _logger.info("Initializing preprocessors are completed!")
