import argparse
import json
import logging
import os
import pickle
import shutil
from typing import Dict

from housing_model.preprocessing.encoder import Encoder, ToList, Lowercase, CategoricalFeature, WhitespaceTokenizer, DateTransformer, \
    PositionEncoder, Scale
from housing_model.preprocessing.utils import generate_files_distributed

_logger = logging.getLogger(__name__)


def build_feature_template(artifact_dir):
    feature_template = {
        'ml_num': ('N2447318', 'NO_ML_NUM', Encoder([ToList()], 'str', 1)),
        'house_type_name': (
            'Detached', 'NoType',
            Encoder(
                [
                    Lowercase(),
                    ToList(),
                    CategoricalFeature(os.path.join(artifact_dir, 'type'), 20)
                ],
                'int32',
                1
            )
        ),
        'house_style': (
            '2-Storey', 'NoStyle',
            Encoder(
                [
                    Lowercase(),
                    ToList(),
                    CategoricalFeature(os.path.join(artifact_dir, 'style'), 20)
                ],
                'int32',
                1
            )
        ),
        'address': (
            '128 Larratt Lane', 'no address',
            Encoder(
                [
                    Lowercase(),
                    WhitespaceTokenizer(),
                    CategoricalFeature(os.path.join(artifact_dir, 'address'), 20000)
                ],
                'int32',
                1
            )
        ),
        'community_name': (
            'Westbrook', 'NoCommunity',
            Encoder(
                [
                    Lowercase(),
                    ToList(),
                    CategoricalFeature(os.path.join(artifact_dir, 'community'), 1000)
                ],
                'int32',
                1
            )
        ),
        'municipality_name': (
            'Richmond Hill', 'No Municipal',
            Encoder(
                [
                    Lowercase(),
                    ToList(),
                    CategoricalFeature(os.path.join(artifact_dir, 'municipal'), 100)
                ],
                'int32',
                1
            )
        ),
        'bedroom': (
            4, -1,
            Encoder(
                [
                    ToList(),
                    CategoricalFeature(os.path.join(artifact_dir, 'bedroom'), 10)
                ],
                'int32',
                1
            )
        ),
        'bedroom_plus': (
            1, -1,
            Encoder(
                [
                    ToList(),
                    CategoricalFeature(os.path.join(artifact_dir, 'bedroom_plus'), 10)
                ],
                'int32',
                1
            )
        ),
        'washroom': (
            2, -1,
            Encoder(
                [
                    ToList(),
                    CategoricalFeature(os.path.join(artifact_dir, 'washroom'), 10)
                ],
                'int32',
                1
            )
        ),
        'parking': {
            "total": (
                2, -1,
                Encoder(
                    [
                        ToList(),
                        CategoricalFeature(os.path.join(artifact_dir, 'total_parking'), 10)
                    ],
                    'int32',
                    1
                )
            ),
            "garage_type": (
                'Built-In', 'No-Garage-Type',
                Encoder(
                    [
                        Lowercase(),
                        ToList(),
                        CategoricalFeature(os.path.join(artifact_dir, 'municipal'), 100)
                    ],
                    'int32',
                    1
                )
            ),
            "garage": (
                2, -1,
                Encoder(
                    [
                        ToList(),
                        CategoricalFeature(os.path.join(artifact_dir, 'total_garage'), 10)
                    ],
                    'int32',
                    1
                )
            ),
            "parking_type": (
                None, 'No-Parking-Type',
                Encoder(
                    [
                        Lowercase(),
                        ToList(),
                        CategoricalFeature(os.path.join(artifact_dir, 'municipal'), 100)
                    ],
                    'int32',
                    1
                )
            ),
            "parking": (
                2, -1,
                Encoder(
                    [
                        ToList(),
                        CategoricalFeature(os.path.join(artifact_dir, 'parking'), 20)
                    ],
                    'int32',
                    1
                )
            ),
        },
        "date_start": (
            "2012-08-24",
            "2000-01-01",
            Encoder([
                DateTransformer('%Y-%m-%d', '2000-01-01'),
                PositionEncoder(100, 16 * 3650 ** 2)  # 40 years
            ], 'float32', 1)
        ),
        "date_end": (
            "2012-09-02",
            "2000-01-01",
            Encoder([
                DateTransformer('%Y-%m-%d', '2000-01-01'),
                PositionEncoder(100, 16 * 3650 ** 2)
            ], 'float32', 1)
        ),
        'price_int': (
            739900,
            0,
            Encoder([
                Scale(1 / 1e6),
                PositionEncoder(100, 20 ** 2)
            ], 'float32', 1)
        ),
        'price_sold_int': (
            739900,
            0,
            Encoder([
                ToList()
            ], 'int32', 1)
        ),
        'map': {
            'lat': (
                43.8934482,
                0,
                Encoder([
                    Scale(100),
                    PositionEncoder(100, 1e8)
                ], 'float32', 1)
            ),
            'lon': (
                -79.4628076,
                0,
                Encoder([
                    Scale(100),
                    PositionEncoder(100, 1e8)
                ], 'float32', 1)
            )
        },
        'house_area': {
            'estimate': (
                None,
                0,
                Encoder([
                    Scale(1e-3),
                    PositionEncoder(100, 100 ** 2)
                ], 'float32', 1)
            )
        },
        'land': {
            'front': (
                39,
                0,
                Encoder([
                    Scale(1/10),
                    PositionEncoder(100, 100 ** 2)
                ], 'float32', 1)
            ),
            'depth': (
                131,
                0,
                Encoder([
                    Scale(1 / 10),
                    PositionEncoder(100, 100 ** 2)
                ], 'float32', 1)
            )
        },
    }
    return feature_template


def get_features(a_listing: dict, faeture_template: dict):
    features = {}
    for feature_name, value in faeture_template.items():
        if isinstance(value, dict):
            features.update(get_features(a_listing[feature_name], value))
        else:
            _, default_value, encoder = value
            if isinstance(a_listing, dict) or a_listing is None:
                feature_value = default_value
                if a_listing is not None and a_listing.get(feature_name) is not None:
                    feature_value = a_listing.get(feature_name)

                try:
                    features[feature_name] = encoder(feature_value)
                except (AssertionError, ValueError) as e:
                    raise ValueError(f'Cannot encode {feature_name} with value of {feature_value}') from e
            else:
                raise ValueError(a_listing)
    return features


def get_encoders(feature_template: dict, encoders: Dict[str, Encoder]):
    for feature_name, value in feature_template.items():
        if isinstance(value, dict):
            get_encoders(value, encoders)
        else:
            _, _, encoder = value
            assert feature_name not in encoders
            encoders[feature_name] = encoder


def data_generator(in_file_path: str,feature_template: dict):
    err = 0
    total_cnt = 0
    with open(in_file_path) as f_in:
        for l in f_in:
            total_cnt += 1
            a_listing = json.loads(l)
            try:
                features = get_features(a_listing, feature_template)
                yield features
            except ValueError as e:
                _logger.exception(f'Cannot convert {l}')
                err += 1
    _logger.info(f'Cannot convert {err} listings, out of {total_cnt} at the second pass')


def first_pass(in_file_path: str,feature_template: dict) -> Dict[str, Encoder]:
    err = 0
    total_cnt = 0
    with open(in_file_path) as f_in:
        for l in f_in:
            total_cnt += 1
            a_listing = json.loads(l)
            try:
                get_features(a_listing, feature_template)
            except ValueError:
                _logger.exception(f'Cannot convert {l}')
                err += 1
    _logger.info(f'Cannot convert {err} listings, out of {total_cnt} at the first pass')

    encoders = {}
    get_encoders(feature_template, encoders)
    for an_encoder in encoders.values():
        an_encoder.save()
    return encoders


def get_schema(encoders: Dict[str, Encoder]) -> Dict[str, Dict[str, str]]:
    schema = {}
    for feature_name, encoder in encoders.items():
        schema[feature_name] = {'dtype': encoder.dtype, 'dim': encoder.dim}
    return schema


def main(input_file: str, output_dir: str):
    num_shards = 10

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    artifact_dir = os.path.join(output_dir, 'artifact')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    feature_template = build_feature_template(artifact_dir)

    encoders = first_pass(input_file, feature_template)

    schema = get_schema(encoders)
    with open(os.path.join(artifact_dir, 'schema.pkl'), 'wb') as fout:
        pickle.dump(schema, fout)

    generate_files_distributed(
        data_generator(input_file, feature_template),
        'house_prediction',
        os.path.join(output_dir, 'tf_data'),
        num_shards
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(**vars(args))
