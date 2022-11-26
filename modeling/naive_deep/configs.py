from dataclasses import dataclass, field
from datetime import datetime
from typing import Set

import tensorflow as tf
from dataclasses_json import DataClassJsonMixin, config


@dataclass
class HyperParams(DataClassJsonMixin):
    embedding_size: int
    num_dense: int
    num_feature_dense: int
    bit_loss_weight: float


@dataclass
class ArchitectureParams(DataClassJsonMixin):
    float_features: Set[str]
    num_bits: int = 32
    price_feature_name: str = "sold_price"
    bits_feature_name: str = "bits"

    @staticmethod
    def from_dataset(train_ds: tf.data.Dataset):
        input_features = set(train_ds.element_spec.keys())
        input_features.remove("metadata")
        input_features.remove("sold_price")
        return ArchitectureParams(input_features)


@dataclass
class EarlyStoppingSetting(DataClassJsonMixin):
    min_delta: float = 0
    patience: int = 100
    verbose: int = 0
    mode: str = "auto"
    restore_best_weights: bool = True


@dataclass
class DatasetSpec(DataClassJsonMixin):
    start: datetime = field(
        metadata=config(
            encoder=lambda date: date.strftime("%Y-%m-%d"),
            decoder=lambda str_date: datetime.strptime(str_date, "%Y-%m-%d"),
        )
    )
    end: datetime = field(
        metadata=config(
            encoder=lambda date: date.strftime("%Y-%m-%d"),
            decoder=lambda str_date: datetime.strptime(str_date, "%Y-%m-%d"),
        )
    )


@dataclass
class DatasetsSpec(DataClassJsonMixin):
    train: DatasetSpec
    dev: DatasetSpec
    test: DatasetSpec


@dataclass
class TrainParams(DataClassJsonMixin):
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping: EarlyStoppingSetting = EarlyStoppingSetting()


@dataclass
class ModelParams(DataClassJsonMixin):
    hyper_params: HyperParams
    arc_params: ArchitectureParams
