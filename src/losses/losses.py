import tensorflow as tf
import math

from tensorflow.python.ops import math_ops


@tf.function
def haf_loss(y_true, y_pred, saliency_maps, reg):
    loss = tf.reduce_mean(tf.reduce_sum(math_ops.pow(y_pred - y_true, 2), axis=1), axis=0)
    reg_loss = 0

    for saliency_map in saliency_maps:
        ti = 1 / math.sqrt(saliency_map.shape[2] * saliency_map.shape[3])
        l1_saliency_map = tf.norm(saliency_map, ord=1)
        reg_loss += ti * l1_saliency_map

    return loss + reg * reg_loss
