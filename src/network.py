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
    
    def GD(training_data, epochs, batch_zise,lr):
        n = len(training_data)
        for i in epochs:
            batch_data = [
                training_data[d:d+batch_zise]
                for d in range(0, n, batch_zise)
            ]
            random.shuffle(training_data)
            for data in batch_data:
                a = data[0]
                z = 
