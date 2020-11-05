import tensorflow as tf
import tensorflow.keras.backend as K


class SaliencyLayer(tf.keras.layers.Layer):

    # there are some difficulties for different types of shapes
    # let's use a 'repeat_count' instead, increasing only one dimension
    def __init__(self, repeat_count=1, **kwargs):
        self.repeat_count = repeat_count
        self.saliency_matrix = None
        super(SaliencyLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        # first, let's get the output_shape
        output_shape = self.compute_output_shape(input_shape)
        weight_shape = (1,) + output_shape[1:]  # replace the batch size by 1
        initializer = tf.initializers.glorot_uniform
        self.saliency_matrix = self.add_weight(name='saliency_matrix',
                                      shape=weight_shape,
                                      initializer=initializer(),
                                      trainable=True,
                                      dtype=tf.float32)

        super(SaliencyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    # here, we need to repeat the elements before multiplying
    def call(self, x):

        if self.repeat_count > 1:
            # we add the extra dimension:
            x = K.expand_dims(x, axis=1)

            # we replicate batch_size the elements
            x = K.repeat_elements(x, rep=self.repeat_count, axis=1)

        # multiply
        return tf.math.multiply(x, self.saliency_matrix)

    # make sure we compute the output shape according to what we did in "call"
    def compute_output_shape(self, input_shape):

        if self.repeat_count > 1:
            return (input_shape[0], self.repeat_count) + input_shape[1:]
        else:
            return input_shape
