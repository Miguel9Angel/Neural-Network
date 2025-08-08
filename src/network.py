import numpy as np
import math
import random
import os
class Network():
    def __init__(self, layers, seed=42):
        np.random.seed(seed)
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def dev_sig(self, a):
        return a*(1-a)

    def output_error(self, a, y):
        return (a - y) * self.dev_sig(a)
    
    def SGD(self, training_data, epochs=10, batch_size=32, lr=0.1):
        n = len(training_data)
        for _ in range(epochs):
            random.shuffle( training_data )
            batches = [ training_data [i:i+ batch_size ] for i in range(0, n, batch_size)]
            for batch in batches:
                X_batch = np.vstack([x for x, _ in batch])
                y_batch = np.vstack([y for _, y in batch])
                activations = self.feedforward(X_batch)
                grads_w, grads_b = self.backpropagation(activations, y_batch)
                self.update_w_b(grads_w, grads_b, lr, X_batch.shape[0])
            
    
    def feedforward(self, X):
        activations = [X]
        a = X
        for W, b in zip(self.weights, self.biases):
            z = np.dot(a, W)+b
            a = self.sigmoid(z)
            activations.append(a)
        return activations
    
    def backpropagation(self, activations, y):
        m = y.shape[0]
        n = len(self.weights)
        grads_w = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        delta = (activations[-1]-y)*self.dev_sig(activations[-1])
        grads_w[-1] = np.dot(activations[-2].T, delta)
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for idx in range(n-2, -1,-1):
            delta = np.dot(delta, self.weights[idx+1].T)*self.dev_sig(activations[idx+1])
            grads_w[idx] = np.dot(activations[idx].T, delta)
            grads_b[idx] = np.sum(delta, axis=0, keepdims=True)
        return grads_w, grads_b

    def update_w_b(self, grads_w, grads_b, lr, batch_size):
        for i in range(len(self.weights)):
            self.weights[i] -= (lr/batch_size)*grads_w[i]
            self.biases[i] -= (lr/batch_size)*grads_b[i]
        
    def predict(self, X):
        X = self._ensure_2d(X)
        return self.feedforward(X)[-1]

    def predict(self, test_data, threshold=0.5):
        X, y = zip(*test_data)
        activations = self.feedforward(X)
        return y, activations[-1]

    def evaluate(self, test_data, threshold=0.5):
        X = np.vstack([x for x,_ in test_data])
        y = np.vstack([y for _,y in test_data])
        preds = self.predict(X)
        preds_bin = (preds >= threshold).astype(int)
        acc = np.mean(preds_bin == y)
        return acc, preds
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_archivo = os.path.join(BASE_DIR, "../data/xor_test.csv")
data = np.loadtxt(ruta_archivo, delimiter=',', dtype=int)
X = data[:, :-1]
y = data[:, -1] 

dataset = [(X[i], [y[i]]) for i in range(len(X))]

train_size = int(len(dataset) * 0.8)
train_data = dataset[:train_size]
test_data = dataset[train_size:]

network1 = Network([2,3,1])
network1.SGD(train_data, 10, 25, 0.25)
output = network1.evaluate(test_data)
print(output)
