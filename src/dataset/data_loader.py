from datetime import date

import tensorflow as tf

from torchvision import transforms
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

from src.dataset.dataset import CustomDataset

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CustomDataLoader:
    def __init__(self, train_dir, image_size=(224, 224), batch_size=16):
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset = None

    def load(self, resnet50):

        pre_softmax_feature_extractor = tf.keras.Model(inputs=resnet50.inputs, outputs=resnet50.layers[-2].output)

        data = CustomDataset(root_dir=self.train_dir,
                             transform=transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))

        images = []
        features = []

        for id, (image, filename) in enumerate(data):
            print(filename)
            image = tf.convert_to_tensor(image)
            image = tf.reshape(image, shape=(image.shape[1], image.shape[2], image.shape[0]))
            images.append(image)

            image = tf.expand_dims(image, axis=0)
            pre_softmax_feature = pre_softmax_feature_extractor.predict(image)
            features.append(pre_softmax_feature[0])

            if id > 5:
                break

        dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(images),
             tf.data.Dataset.from_tensor_slices(features)
             )
        )

        dataset = dataset.batch(batch_size=self.batch_size)
        self.dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        print("The data loader has loaded the dataset.")
