from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf


# 元のCifar-10の画像サイズは32 x 32.
IMAGE_SIZE = 24

# 扱うクラス数
NUM_CLASSES = 10

# 学習回数
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

# 評価回数
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """
    CIFAR10のデータを読み込む

    Args:
        filename_queue: A queue of strings with the filenames to read from.
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # CIFAR-10データのサイズ
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height + result.width * result.depth

    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.ini32)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])

    result.unit8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """
    画像、ラベルのキューを作成する
    """
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
           [image, label],
           batch_size=batch_size,
           num_threads=num_preprocess_threads,
           capacity=min_queue_examples + 3 * batch_size)

    # 学習中の画像を保存
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def disorted_inputs(data_dir, batch_size):
    """
    学習時にデータを分散して読ませる
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 読み込むファイルを持つキュー
    filename_queue = tf.train.string_input_producer(filenames)

    # 
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.unit8image, tf.float32)

    height = IMAGE
        
