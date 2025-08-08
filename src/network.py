import numpy as np
import math
import random
import os
class Network():
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.rand(1, y) for y in layers[1:]]

    def sigmoid(self, z):
        return 1 / (1+np.exp(z))
    
    def dev_sig(self, a):
        return a*(1-a)

    def output_error(self, a, y):
        return (a - y) * self.dev_sig(a)
    
    def SGD(self, training_data, epochs, batch_size, lr):
        n = len(training_data)
        for _ in range(epochs):
            random.shuffle( training_data )
            batches = [ training_data [i:i+ batch_size ] for i in range(0, n,
            batch_size )]
            for batch in batches:
                for x, y in batch:
                    activations = self.feedforward(x)
                    errors = self.backpropagate_error(activations, y)
                    n_data = len(x)
                    self.GD(errors, activations, lr, n_data)
            
    
    def feedforward(self, X):
        activations = [X]
        for i in range(0, len(self.layers)-1):
            z = np.dot(activations[i], self.weights[i])+self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        return activations
    
    def backpropagate_error(self, activations, y):
        output = activations[len(activations)-1]
        errors = [(output - y) * self.dev_sig(output)]
        n_layers = len(self.layers)
        for idx in range(2, n_layers):
            error = np.dot(errors[-idx+1], self.weights[-idx+1].T)*self.dev_sig(activations[-idx])
            errors.insert(0, error)
        return errors

    def GD(self, errors, activations, lr, n_data):
        for idx in range(0, len(self.layers)-1):
            self.weights[idx] = self.weights[idx] - (lr/n_data)*np.sum(np.dot(errors[idx], activations[idx].T))
            self.biases[idx] = self.biases[idx] - (lr/n_data)*np.sum(errors[idx])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_archivo = os.path.join(BASE_DIR, "../data/xor_test.csv")
data = np.loadtxt(ruta_archivo, delimiter=',', dtype=int)
X = data[:, :-1]
y = data[:, -1] 

dataset = [(X[i], [y[i]]) for i in range(len(X))]

train_size = int(len(dataset) * 0.8)
train_data = dataset[:train_size]
test_data = dataset[train_size:]

network1 = Network([2,5,5,1])
network1.SGD(train_data, 10, 25, 0.25)

