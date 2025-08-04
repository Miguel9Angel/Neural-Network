import numpy as np
import matplotlib as mpl
import math
import random

class Network():
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.rand(1, y) for y in layers[1:]]

    def sigmoid(z):
        return 1 / (1+math.exp(z))
    
    def GD(x_train_data, y_train_data, epochs, batch,lr):
        
        for i in epochs:
            iter = int(x_train_data/batch)
            for j in iter:
