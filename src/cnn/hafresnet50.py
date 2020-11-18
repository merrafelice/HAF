import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from bin.utils.timethis import timethis
from src.cnn.hafmodel import HAFModel
from src.layers.utils import insert_saliency_layers
from src.losses.losses import haf_loss_single
import copy

AUTOTUNE = tf.data.experimental.AUTOTUNE


class HAFResNet50Model(HAFModel):
    def __init__(self, base_model, loss_sc=False, reg=1.0):
        super().__init__(base_model=base_model)
        self.loss_sc = loss_sc
        self.reg = reg

    def insert_saliency_layers(self, layers, is_after=1):
        """
        layers: Where to put the saliency layers
        """
        layer_names = ['saliency_{0}'.format(i) for i in range(len(layers))]

        # self.haf_model, self.tis = insert_saliency_layers(self.base_model, layers, layer_names=layer_names,
        #                                                   position='after' if is_after else 'before')

        self.haf_model, self.tis = insert_saliency_layers(self.base_model, layers, self.reg,
                                                          position='after' if is_after else 'before')

        # self.haf_model.layers[-1].activation = tf.keras.layers.Activation(activation='linear')

    def haf_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(math_ops.pow(y_pred - y_true, 2))
        reg_loss = 0

        for i, saliency_map in enumerate(self.haf_model.trainable_variables):
            reg_loss += self.tis[i] * tf.norm(saliency_map, ord=1)

        return loss  # + self.reg * reg_loss

    @timethis
    def train(self, data_loader, lr=0.05, epochs=30, train_dir=None, path_saved_smaps=None):
        """
        lr: learning rate
        epochs: number of epochs
        reg: regularization factor (lambda) in eq.17 of HAF paper
        """

        optimizer = tf.keras.optimizers.Adam(lr=lr)

        print("ELABORATE ALL THE IMAGES")
        data_map = data_loader.data_map.batch(1)

        for imagine, ground_score, ground_class, filename in data_map:
            # Re-initialize the trainable-weights for the new image
            print('\tImage: {0}'.format(filename[0].numpy().decode('utf-8')))

            self.reinit_trainable_variables()

            best_loss = np.inf
            M = []  # The list where we will store the best model

            for iteration in range(1, 31):

                with tf.GradientTape() as tape:
                    # make a prediction using the model and then calculate the loss
                    full_score_class = self.haf_model(imagine)

                    ll, loss = haf_loss_single(self.tis, ground_class, ground_score, full_score_class,
                                           self.haf_model.trainable_variables, reg=self.reg)

                # calculate the gradients using our tape and then update the model weights
                grads = tape.gradient(loss, self.haf_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.haf_model.trainable_variables))

                # Project Gradient Descent
                for i, trainable_variable in enumerate(self.haf_model.trainable_variables):
                    self.haf_model.trainable_variables[i].assign(tf.clip_by_value(trainable_variable, clip_value_min=0,
                                                                                  clip_value_max=np.inf))

                if ll < best_loss:
                    best_loss = ll
                    M = copy.deepcopy(self.haf_model.trainable_variables)

                print("\t\t\tIteration: {} Loss: {} Diff: {}".format(iteration, loss, ll))

            ## PRINT the Best M and the HAF
            self.plot_and_save_saliency_maps_for_an_image(M, filename[0].numpy().decode('utf-8'), train_dir,
                                                          path_saved_smaps, show=False,
                                                          save=True)

        print("Train Ended")
