import re
from builtins import object

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import scipy.ndimage as ndimage
from bin.utils.IO import save_obj, load_obj
from src.dataset.data_loader import CustomDataLoader
from src.dataset.utils import read_image
from src.layers.utils import insert_saliency_layers
from src.plot.utils import deprocess_img
import matplotlib.cm as cm

np.random.seed(2006)


class HAFModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.haf_model = None
        self.losses = []  # The loss evolution at each batch iteration
        self.haf_mask = None  # The final saliency mask (M of eq. 17 in teh paper)

    def make_base_model_untrainable(self):
        """
        Make the base model untrainable in order to train the saliency maps
        """
        for layer in self.base_model.layers:
            layer.trainable = False

    def insert_saliency_layers(self, layers):
        """
        layers: Where to put the saliency layers
        """
        pass

    def train(self, data_loader: CustomDataLoader, lr=0.05, epochs=30, reg=0.5):
        """
        lr: learning rate
        epochs: number of epochs
        reg: regularization factor (lambda) in eq.17 of HAF paper
        """
        pass

    def plot_loss(self, path_weights, show=False, save=True):
        """
        Plot the loss evolution across iterations
        """
        plt.plot(range(len(self.losses)), self.losses)
        plt.ylabel('Loss')
        plt.xlabel('Number of Iterations')
        if show:
            plt.show()
        if save:
            plt.savefig(path_weights + 'losses.jpg')

    def get_model_with_saliency_output(self, new_layer):
        """
        Build teh model with the correct output (saliency map)
        """
        for id, layer in enumerate(self.haf_model.layers):
            if re.match(new_layer, layer.name):
                break

        return tf.keras.Model(inputs=self.haf_model.inputs, outputs=self.haf_model.layers[id + 1].output)

    def plot_and_save_saliency_maps(self, new_layers, train_dir, path_saved_smaps, show=True, save=False):
        """
        Saliency Mask Visualization inspired by https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
        """
        filename_images = np.random.choice(os.listdir(train_dir), 2)

        for filename_image in filename_images:

            image = read_image(train_dir + filename_image)

            final_saliency = []

            for new_layer in new_layers:
                # redefine model to output right after the first hidden layer
                model = self.get_model_with_saliency_output(new_layer)

                # get saliency map for first hidden layer
                saliency_map = model.predict(image)

                saliency_map = np.mean(saliency_map, axis=-1)  # Channel Mean
                saliency_map = np.resize(saliency_map,
                                         new_shape=(
                                             224,
                                             224))  # Resize up-samples high-level attributions to the full image size
                saliency_map = np.divide(saliency_map, np.max(saliency_map))
                final_saliency.append(saliency_map)

                plt.imshow(deprocess_img(image))

                gaus = ndimage.gaussian_filter(saliency_map, sigma=5)
                plt.imshow(gaus, alpha=.7)

                # Create Image Directory
                dir_image = path_saved_smaps + filename_image.split('.')[0]
                os.makedirs(dir_image, exist_ok=True)
                obj_name = '{0}/sm_{1}'.format(dir_image, model.layers[-1].name)
                save_obj(saliency_map, obj_name)

                if show:
                    plt.show()
                if save:
                    plt.savefig(obj_name + '.jpg')

            # Final Saliency HAF

            plt.imshow(deprocess_img(image))

            plt.imshow(deprocess_img(image))
            final_saliency = np.mean(final_saliency, axis=0)
            final_gaus = ndimage.gaussian_filter(final_saliency, sigma=5)
            plt.imshow(final_gaus, alpha=.7)
            plt.savefig(obj_name + '_HAF.jpg')

        print('Completed the Generation of Saliency Plots')

    def grid_plot_and_save_saliency_maps(self, new_layers, train_dir, path_saved_smaps, show=True, save=False):
        """
        Saliency Mask Visualization inspired by https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
        """
        filename_images = np.random.choice(os.listdir(train_dir), 2)

        for filename_image in filename_images:

            image = read_image(train_dir + filename_image)

            for new_layer in new_layers:
                # redefine model to output right after the first hidden layer
                model = self.get_model_with_saliency_output(new_layer)

                # get saliency map for first hidden layer
                saliency_map = model.predict(image)

                # plot all the saliency maps in a square
                square = int(np.sqrt(saliency_map.shape[-1]))
                ix = 1
                for _ in range(square):
                    for _ in range(square):
                        # specify subplot and turn of axis
                        ax = plt.subplot(square, square, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        gaus = ndimage.gaussian_filter(saliency_map[0, :, :, ix - 1], sigma=5)
                        plt.imshow(gaus)  # , cmap='gray')
                        ix += 1

                # Create Image Directory
                dir_image = path_saved_smaps + filename_image.split('.')[0]
                os.makedirs(dir_image, exist_ok=True)
                obj_name = '{0}/sm_{1}'.format(dir_image, model.layers[-1].name)
                save_obj(saliency_map, obj_name)

                if show:
                    plt.show()
                if save:
                    plt.savefig(obj_name + '_grid.jpg')

        print('Completed the Generation of Saliency Plots')

    def save_trainable_variables(self, path_weights):
        save_obj(self.haf_model.trainable_variables, path_weights + 'trainable_vars')

    def restore_trainable_variables(self, path_weights):
        try:
            trainable_variables = load_obj(path_weights + 'trainable_vars')
            self.haf_model.trainable_variables = trainable_variables
        except:
            print('Error in the trainable variable loading\n*** Random Initialization ***')