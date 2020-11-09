import os
import numpy as np
import tensorflow as tf
import pandas as pd

from datetime import date
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from bin.utils.timethis import timethis
from src.dataset.utils import read_image

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CustomDataLoader:
    def __init__(self, train_dir, image_size=(224, 224), batch_size=16, window=0):
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset = None
        self.window = np.inf if window == 0 else window

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

        dataset = dataset.batch(batch_size=self.batch_size)
        self.dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        print("The data loader has loaded the dataset.")
