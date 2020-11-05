import tensorflow as tf
import math

from tensorflow.python.ops import math_ops


def haf_loss(y_true, y_pred, saliency_masks, reg):
    loss = tf.reduce_mean(tf.reduce_sum(math_ops.pow(y_pred - y_true, 2), axis=1), axis=0)
    reg_loss = 0

    for saliency_mask in saliency_masks:
        ti = 1/math.sqrt(saliency_mask.shape[2]*saliency_mask.shape[3])
        l1_saliency_mask = tf.norm(saliency_mask, ord=1)
        reg_loss += ti*l1_saliency_mask

    return loss + reg*reg_loss