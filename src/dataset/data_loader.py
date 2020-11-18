import os
from functools import partial

import numpy as np
import tensorflow as tf
import pandas as pd

from datetime import date
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from bin.utils.timethis import timethis
from src.dataset.utils import read_image

AUTOTUNE = tf.data.experimental.AUTOTUNE


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

# Create a dictionary describing the features.


def _parse_image_function(example_proto):
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'score': tf.io.FixedLenFeature([1000], tf.float32),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    #
    return tf.cast(example["image_raw"], tf.float32), tf.cast(example["score"], tf.float32), tf.cast(example["label"],
                                                                                                     tf.int32)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def serialize_example(image, score, image_class):
    # image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'label': _int64_feature(image_class),
        'score': tf.train.Feature(float_list=tf.train.FloatList(value=score)),
        'image_raw': _bytes_feature(image.numpy().tostring()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


class CustomDataLoader:
    def __init__(self, train_dir, image_size=(224, 224), batch_size=16, window=0, resnet50=None):
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset = None
        self.window = np.inf if window == 0 else window

        self.resnet50 = resnet50
        self.resnet50.layers[-1].activation = tf.keras.layers.Activation(activation='linear')

    def pp(self, filename):
        image = read_image(filename)
        img = tf.expand_dims(image, axis=0)
        score = self.resnet50(img, training=False)
        image_classe = tf.math.argmax(score[0])
        return image, score[0], image_classe, filename

    def pre_process(self, filename):
        image, score, image_classe = tf.py_function(self.pp, (self.train_dir + filename,),
                                                    (tf.float32, tf.float32, tf.int64,))
        return image, score, image_classe, filename

    @timethis
    def load_files(self):
        self.data_map = tf.data.Dataset.from_tensor_slices(sorted(os.listdir(self.train_dir)))
        self.data_map = self.data_map.map(self.pre_process, AUTOTUNE)

    @timethis
    def load(self, resnet50):

        resnet50.layers[-1].activation = tf.keras.layers.Activation(activation='linear')

        images = []
        scores = []
        image_classes = []

        for id, filename in enumerate(os.listdir(self.train_dir)):

            image = read_image(self.train_dir + filename)
            images.append(image[0])
            score = resnet50.predict(image)
            scores.append(score[0])
            image_classes.append(tf.math.argmax(score[0]))

            if id > self.window:
                break

        dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(images),
             tf.data.Dataset.from_tensor_slices(scores),
             tf.data.Dataset.from_tensor_slices(image_classes)
             )
        )

        ##### SAVE ########
        # record_file = 'images.tfrecords'
        # with tf.io.TFRecordWriter(record_file) as writer:
        #     for image, score, image_class in dataset:
        #         tf_example = serialize_example(image, score, image_class)
        #         writer.write(tf_example.SerializeToString())
        #######################

        dataset = dataset.batch(batch_size=self.batch_size)
        self.dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        print("The data loader has loaded the dataset.")

    @timethis
    def read_tfrecord(self):
        record_file = 'images.tfrecords'

        dataset = tf.data.TFRecordDataset(record_file)
        parsed_dataset = dataset.map(partial(_parse_image_function), num_parallel_calls=8)

        # raw_example = next(iter(dataset))
        # parsed = tf.train.Example.FromString(raw_example.numpy())

        self.dataset = parsed_dataset.shuffle(buffer_size=self.batch_size * 8).batch(self.batch_size).prefetch(
            buffer_size=AUTOTUNE)
