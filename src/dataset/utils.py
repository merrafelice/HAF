from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image


def read_image(path):
    img = Image.open(path.numpy())
    img = img.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')
    x = np.array(img)
    x = preprocess_input(x)
    return x


def old_read_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
