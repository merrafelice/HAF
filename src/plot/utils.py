import numpy as np
import tensorflow.keras.backend as K


def normalize(array, min_value=0., max_value=1.):
    """Normalizes the numpy array to (min_value, max_value)

    Args:
        array: The numpy array
        min_value: The min value in normalized array (Default value = 0)
        max_value: The max value in normalized array (Default value = 1)

    Returns:
        The array normalized to range between (min_value, max_value)
    """
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + K.epsilon())
    return (max_value - min_value) * normalized + min_value


def deprocess_img(processed_img):
    """
    Come back from caffe mode
    """
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x
