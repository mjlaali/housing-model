import abc
import logging
import os
import pickle
from collections import Counter
from datetime import datetime
from typing import List, Union

import numpy as np

_logger = logging.getLogger(__name__)


class Transformation(abc.ABC):

    @abc.abstractmethod
    def analyze(self, raw: object) -> object:
        pass

    @abc.abstractmethod
    def process(self, raw: object) -> object:
        pass

    def save(self):
        pass


class StatelessTransformation(Transformation, abc.ABC):
    def analyze(self, raw: object) -> object:
        return self.process(raw)

    def save(self):
        pass


class WhitespaceTokenizer(StatelessTransformation):
    def process(self, raw: str) -> list:
        return raw.split(' ')


class CategoricalFeature(Transformation):
    UNK = 'unk'

    def __init__(self, vocab_file, num_values):
        self._vocab_file = vocab_file
        self._token_id = {self.UNK: 0}
        self._num_values = num_values
        self._if_analyze = not os.path.exists(vocab_file)
        if self._if_analyze:
            self._vocabs = Counter()
        else:
            with open(self._vocab_file, 'rb') as fin:
                self._vocabs = pickle.load(fin)
            _logger.warning(f'The vocab file {vocab_file} already exist, hence, vocabs will not be computed again')

    def analyze(self, raw: list) -> list:
        res = []
        for a_token in raw:
            if self._if_analyze:
                self._vocabs[a_token] += 1
            res.append(0)
        return res

    def process(self, raw: list) -> list:
        res = []
        for a_token in raw:
            token_id = self._token_id.get(a_token)
            if token_id is None:
                token_id = self._token_id[self.UNK]
            res.append(token_id)
        return res

    def save(self):
        if self._if_analyze:
            with open(self._vocab_file, 'wb') as fout:
                pickle.dump(self._vocabs, fout)
        total_tokens = sum(self._vocabs.values())
        considered_token = 0
        for token, freq in self._vocabs.most_common(self._num_values):
            considered_token += freq
            self._token_id[token] = len(self._token_id)
        _logger.info(f'{self._vocab_file} covers {considered_token/total_tokens:.2f} of tokens.')


class ToList(StatelessTransformation):
    def process(self, raw: object) -> list:
        return [raw]


class Lowercase(StatelessTransformation):
    def process(self, raw: str) -> str:
        return raw.lower()


class DateTransformer(StatelessTransformation):
    def __init__(self, template_format: str, base: str):
        self._template_format = template_format
        self._base = datetime.strptime(base, template_format).date()

    def process(self, raw: str) -> int:
        delta = datetime.strptime(raw, self._template_format).date() - self._base
        return delta.days


class Scale(StatelessTransformation):
    def __init__(self, scale: Union[float, int]):
        self._scale = scale

    def process(self, raw: float) -> float:
        return self._scale * raw


class PositionEncoder(StatelessTransformation):
    def __init__(self, dim, scale):
        i = 2 * np.arange(dim) // 2
        self._angle_rates = 1 / np.power(scale, i / np.float32(dim))

    def process(self, raw: int) -> list:
        angle_rads = raw * self._angle_rates

        # apply sin to even indices in the array; 2i
        angle_rads[0::2] = np.sin(angle_rads[0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[1::2] = np.cos(angle_rads[1::2])

        return angle_rads


class Encoder(object):
    def __init__(self, transformations: List[Transformation], dtype: str, dim: int):
        self._transformations = transformations
        self._mode = 'analyze'
        self._dtype = dtype
        self._dim = dim

    def __call__(self, raw_input):
        assert raw_input is not None
        in_val = raw_input
        out_val = None
        for a_transformation in self._transformations:
            op = getattr(a_transformation, self._mode)
            out_val = op(in_val)
            if out_val is None:
                raise ValueError(f'{type(a_transformation)} generates None value for {in_val}')
            in_val = out_val

        try:
            feature_value = np.asarray(out_val, dtype=self._dtype)
        except TypeError as e:
            raise ValueError(f'{out_val} is not compatible with {self._dtype}') from e
        assert len(feature_value.shape) == self._dim
        return feature_value

    def save(self):
        for a_transformation in self._transformations:
            a_transformation.save()
        self._mode = 'process'

    @property
    def dtype(self):
        return self._dtype

    @property
    def dim(self):
        return self._dim
