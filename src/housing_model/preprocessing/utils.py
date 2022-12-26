import shutil

import six
import numpy as np
import tensorflow as tf
import os
import logging

_logger = logging.getLogger(__name__)


def str_feature(v):
    v = [bytes(x, "utf-8") for x in v]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))


def feature_factory(dtype):
    if np.issubdtype(dtype, np.int32):
        return lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    if np.issubdtype(dtype, np.float32):
        return lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
    if np.issubdtype(dtype, np.str_):
        return str_feature
    raise ValueError(f"{dtype} is not supported")


def to_example(dictionary):
    features = {}
    for (k, v) in six.iteritems(dictionary):
        if v is None or len(v) == 0:
            raise ValueError("Empty generated field: %s" % str((k, v)))

        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        feature_gen = feature_factory(v.dtype)
        features[k] = feature_gen(v)

    return tf.train.Example(features=tf.train.Features(feature=features))


def sharded_name(base_name, shard, total_shards):
    """Copied from tensor2tensor github repository"""
    return "%s-%.5d-of-%.5d" % (base_name, shard, total_shards)


def generate_files_distributed(
    generator, output_name, output_dir, num_shards=1, max_cases=None, task_id=0
):
    """
    generate_files but with a single writer writing to shard task_id.
    Copied from tensor2tensor github repository
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert task_id < num_shards
    output_filename = sharded_name(output_name, task_id, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    _logger.info("Writing to file %s", output_file)
    writer = tf.io.TFRecordWriter(output_file)

    counter = 0
    for case in generator:
        if counter % 100000 == 0:
            _logger.info("Generating case %d for %s." % (counter, output_name))
        counter += 1
        if max_cases and counter > max_cases:
            break
        example = to_example(case)
        writer.write(example.SerializeToString())

    writer.close()
    return output_file
