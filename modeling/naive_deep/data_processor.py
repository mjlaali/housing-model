from dataclasses import dataclass

import tensorflow as tf

from housing_model.modeling.naive_deep.configs import ArchitectureParams
from housing_model.modeling.naive_deep.model_builder import num_to_bits


@dataclass
class DatasetProcessor:
    """
    Prepare a data set for a specific model.
    """

    model_params: ArchitectureParams

    def setup_data(self, tf_data: tf.data.Dataset, batch_size: int):
        data = tf_data.map(
            lambda ex: (
                {f_name: ex[f_name] for f_name in self.model_params.float_features},
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
