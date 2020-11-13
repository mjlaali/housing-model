import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

from housing_model.preprocessing.encoder import WhitespaceTokenizer, CategoricalFeature, PositionEncoder, Encoder
import numpy as np

class TestWhitespaceTokenizer(TestCase):
    def test_tokenize(self):
        tokenizer = WhitespaceTokenizer()
        res = tokenizer.analyze('it is a test')
        assert res == ['it', 'is', 'a', 'test']


class TestCategoricalFeature(TestCase):
    def test_analyze(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            with patch('os.path.exists', lambda x: False):
                transformation = CategoricalFeature(tmp_file.name, 10)
            transformation.analyze(['a', 'a', 'a', 'b', 'b', 'c'])
            pickle_dump = MagicMock()
            with patch('pickle.dump', pickle_dump):
                transformation.save()
            res = transformation.process(['a', 'b', 'c'])
            assert len(res) == 3
            assert res == [1, 2, 3]


class TestPositionEncoder(TestCase):
    def test_encode_zero(self):
        encoder = PositionEncoder(10, 100)
        emb = encoder.process(0)
        np.testing.assert_almost_equal(emb[0::2], 0)
        np.testing.assert_almost_equal(emb[1::2], 1)


class TestEncoder(TestCase):
    def test_analyse_mode(self):
        transformation = MagicMock()
        transformation.analyze = MagicMock(return_value=[1, 1, 1])
        transformation.process = MagicMock(return_value=[2, 2, 2])
        transformation.save = MagicMock()

        encoder = Encoder([transformation], 'int32', 1)
        raw_input = 'dummy'
        res = encoder(raw_input)
        assert all(res == [1, 1, 1])

        encoder.save()
        transformation.save.assert_called_once()

        res = encoder(raw_input)
        assert all(res == [2, 2, 2])