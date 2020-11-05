import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from src.dataset.data_loader import CustomDataLoader
from src.layers.utils import insert_saliency_layers


class HAFModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.haf_model = None
        self.losses = []  # The loss evolution at each batch iteration

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

    def plot_loss(self, show=True):
        plt.plot(range(len(self.losses)), self.losses)
        plt.ylabel('Loss')
        plt.xlabel('Number of Iterations')
        if show:
            plt.show()

    def save_trainable_variables(self, path_weights):
        with open(path_weights + 'trainable_vars.pkl', 'wb') as f:
            pickle.dump(self.haf_model.trainable_variables, f)

    def restore_trainable_variables(self, path_weights):
        try:
            with open(path_weights + 'trainable_vars.pkl', 'rb') as f:
                trainable_variables = pickle.load(f)
            self.haf_model.trainable_variables = trainable_variables
        except:
            print('Error in the trainable variable loading\\*** Random Initialization ***')
