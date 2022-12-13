import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import tensorflow as tf
from absl import logging

from housing_model.modeling.naive_deep.configs import ArchitectureParams
from housing_model.modeling.naive_deep.model_builder import num_to_bits


@dataclass
class DatasetProcessor:
    """
    Prepare a data set for a specific model.
    """

    model_params: ArchitectureParams
    preprocessors: Optional[Dict[
        str, tf.keras.layers.experimental.preprocessing.PreprocessingLayer
    ]] = field(init=False, default=None)

    def setup_data(self, tf_data: tf.data.Dataset, batch_size: int):
        preprocessors = self.preprocessors
        if preprocessors is None:
            raise ValueError("Please call setup_preprocessors first.")

        data = tf_data.map(
            lambda ex: (
                {
                    f_name: ex[f_name]
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

    def setup_preprocessors(self, tf_data: tf.data.Dataset, sample_size: int):
        logging.info("Initializing preprocessors ...")
        assert self.preprocessors is None, "Preprocessors is already initialized"
        self.preprocessors = {}

        cached_data = tf_data.take(sample_size)
        for f_name in self.model_params.float_features:
            logging.info(f"Initializing {f_name} ...")
            self.preprocessors[f_name] = tf.keras.layers.Normalization(axis=None)
            self.preprocessors[f_name].adapt(cached_data.map(lambda ex: ex[f_name]))
        logging.info(f"Initialization got completed")

    def add_preprocessors(self, model: tf.keras.models.Model) -> tf.keras.models.Model:
        assert isinstance(model.input, dict), "Please use dictionary for defining the input of the model"
        inputs = {}

        for f_name, input_layer in model.input.items():
            # FIXME: get the shape from the input model
            cloned_input = tf.keras.layers.Input(name=f_name, dtype=input_layer.dtype, shape=())
            if f_name in self.preprocessors:
                cloned_input = self.preprocessors[f_name](cloned_input)
            inputs[f_name] = cloned_input

        # we need to explicitly convert model outputs to dict to avoid keras confuse its type is list.
        inner_model_output = model(inputs)
        outputs = {
            name: tf.keras.layers.Lambda(lambda x: x, name=name)(value)
            for name, value in inner_model_output.items()
        }

        return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="normalized_model")

