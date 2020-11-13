import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
import math
from bin.utils.timethis import timethis
from src.cnn.hafmodel import HAFModel
from src.layers.utils import insert_saliency_layers
from src.losses.losses import haf_loss, haf_loss_single


class HAFResNet50Model(HAFModel):
    def __init__(self, base_model, loss_sc=False):
        super().__init__(base_model=base_model)
        self.loss_sc = loss_sc

    def insert_saliency_layers(self, layers, is_after=1):
        """
        layers: Where to put the saliency layers
        """
        layer_names = ['saliency_{0}'.format(i) for i in range(len(layers))]

        self.haf_model, self.tis = insert_saliency_layers(self.base_model, layers, layer_names=layer_names,
                                                          position='after' if is_after else 'before')

        self.haf_model.layers[-1].activation = tf.keras.layers.Activation(activation='linear')

    def haf_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.reduce_sum(math_ops.pow(y_pred - y_true, 2), axis=1), axis=0)
        reg_loss = 0

        for i, saliency_map in enumerate(self.haf_model.trainable_variables):
            reg_loss += self.tis[i] * tf.norm(saliency_map, ord=1)

        return loss + self.reg * reg_loss

    @timethis
    def train(self, data_loader, lr=0.05, epochs=30, reg=0.5):
        """
        lr: learning rate
        epochs: number of epochs
        reg: regularization factor (lambda) in eq.17 of HAF paper
        """

        optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.reg = reg
        for epoch in range(1, epochs + 1):
            dataset = data_loader.dataset.shuffle(buffer_size=data_loader.batch_size * 8)

            it_loss = []
            for iteration, (imagine, ground_score, ground_class) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    # make a prediction using the model and then calculate the loss
                    full_score_class = self.haf_model(imagine)

                    if self.loss_sc:
                        loss = haf_loss_single(self.tis, ground_class, ground_score, full_score_class,
                                               self.haf_model.trainable_variables, reg=reg)
                        # loss = tf.keras.losses.MeanAbsoluteError(ground_class, ground_score)
                    else:
                        loss = self.haf_loss(ground_score, full_score_class)
                        # loss = tf.keras.losses.MeanSquaredError()
                        # loss = loss(full_score_class, ground_score)

                        # loss2 = haf_loss(ground_score, full_score_class, self.haf_model.trainable_variables, reg=reg)

                it_loss.append(loss)
                # calculate the gradients using our tape and then update the model weights
                grads = tape.gradient(loss, self.haf_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.haf_model.trainable_variables))

                # Project Gradient Descent
                for i, trainable_variable in enumerate(self.haf_model.trainable_variables):
                    self.haf_model.trainable_variables[i].assign(tf.clip_by_value(trainable_variable, clip_value_min=0,
                                                                                  clip_value_max=np.inf))

                if (iteration + 1) % 100 == 0:
                    print("\tEpoch: {} Iteration: {} Loss: {}".format(epoch, iteration, loss))

            print("Epoch: {} Loss: {}".format(epoch, np.mean(it_loss)))

            self.losses.append(np.mean(it_loss))

        print("Train Ended")
