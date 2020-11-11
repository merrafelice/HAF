import re
from builtins import object
import tensorflow.keras.backend as K
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
from src.plot.utils import deprocess_img, normalize
import matplotlib.cm as cm
import cv2

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
            os.makedirs(path_weights, exist_ok=True)
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
        filename_images = np.random.choice(os.listdir(train_dir), 5)

        for i, filename_image in enumerate(filename_images):

            print('{} - Save saliency maps on image {}'.format(i+1, filename_image))

            image = read_image(train_dir + filename_image)

            final_saliency = []

            for new_layer in new_layers:
                # PLOT THE BASE IMAGE
                plt.imshow(deprocess_img(image))

                # redefine model to output right after the first hidden layer
                model = self.get_model_with_saliency_output(new_layer)

                # get saliency map for first hidden layer
                saliency_maps = model.predict(image)

                # Evaluate the Mean over the Axis (e.g., there are 256 on the first saliency layer)
                saliency_map = np.mean(saliency_maps, axis=-1)  # Channel Mean

                # RESIZE
                # saliency_map = cv2.resize(saliency_map[0], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                # This Resize give the effect with squares that we have in the paper
                saliency_map = cv2.resize(saliency_map[0], dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                # NORMALIZE
                saliency_map = np.divide(saliency_map, np.max(saliency_map))
                # saliency_map = normalize(saliency_map)

                # Isolates the Squares as shown in the paper
                saliency_map = np.uint8(cm.jet(saliency_map)[..., 0] * 255) # Probably we need to take the three channel

                # ADD to the FINAL SALIENCY MAP
                final_saliency.append(saliency_map)

                # GAUSSIAN FILTER
                # saliency_map = ndimage.gaussian_filter(saliency_map, sigma=5)

                plt.imshow(saliency_map, alpha=.7)

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
        filename_images = np.random.choice(os.listdir(train_dir), 50)

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
        os.makedirs(path_weights, exist_ok=True)

        save_obj(self.haf_model.trainable_variables, path_weights + 'trainable_vars')
        save_obj(self.haf_model.trainable_weights, path_weights + 'trainable_weights')
        save_obj(self.haf_model.weights, path_weights + 'weights')

    def restore_trainable_variables(self, path_weights):
        try:
            trainable_variables = load_obj(path_weights + 'trainable_vars')
            for i, trainable_variable in enumerate(trainable_variables):
                self.haf_model.trainable_variables[i].assign(trainable_variable)
            # trainable_weights = load_obj(path_weights + 'trainable_weights')
            # for i, trainable_weight in trainable_weights:
            #     self.haf_model.trainable_weights[i].assign(trainable_weight)
            # weights = load_obj(path_weights + 'weights')
            # for i, weight in weights:
            #     self.haf_model.weights[i].assign(weight)
            return 0
        except:
            print('Error in the restoring\n*** Random Initialization ***')
            return 1