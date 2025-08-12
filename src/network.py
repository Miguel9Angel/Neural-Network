import numpy as np
import math
import random
import os
from keras.datasets import mnist

class Network():
    def __init__(self, layers, seed=42, cost='quadratic'):
        np.random.seed(seed)
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        self.cost = cost

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
                grads_w, grads_b = self.backpropagation(activations, y_batch, self.cost)
                self.update_w_b(grads_w, grads_b, lr, X_batch.shape[0])
            
    
    def feedforward(self, X):
        activations = [X]
        a = X
        for W, b in zip(self.weights, self.biases):
            
            z = np.dot(a, W)+b
            a = self.sigmoid(z)
            activations.append(a)
        return activations
    
    def backpropagation(self, activations, y, cost):
        m = y.shape[0]
        n = len(self.weights)
        grads_w = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        if cost == 'quadratic':
            delta = (activations[-1]-y)*self.dev_sig(activations[-1])
        elif cost == 'cross entropy':
            delta = (activations[-1]-y)
        
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
        
    def predict(self, X, y=None):
        if isinstance(X, list) and isinstance(X[0], tuple):
            X = np.vstack([xi for xi, _ in X])
        activations = self.feedforward(X)
        return activations[-1]

    def evaluate(self, test_data, threshold=0.5):
        X = np.vstack([x for x,_ in test_data])
        y = np.vstack([y for _,y in test_data])
        preds = self.predict(X)
        
        if preds.shape[1] == 1:
            preds_bin = (preds >= threshold).astype(int)
            acc = np.mean(preds_bin == y)
        else:
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(preds, axis=1)
            acc = np.mean(y_pred == y_true)
        return acc, preds


def testing_network(datasets):
    for dataset_name in datasets:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ruta_archivo = os.path.join(BASE_DIR, "../data/"+dataset_name)
        data = np.loadtxt(ruta_archivo, delimiter=',', dtype=int)
        X = data[:, :-1]
        y = data[:, -1] 

        dataset = [(X[i], [y[i]]) for i in range(len(X))]

        train_size = int(len(dataset) * 0.8)
        train_data = dataset[:train_size]
        test_data = dataset[train_size:]

        network1 = Network([2, 30, 1])
        network1.SGD(train_data, 10, 10, 0.5)
        accuracy, prediction = network1.evaluate(test_data)
        print('accuracy '+dataset_name )
        print(accuracy)

datasets = ['and_test.csv', 'or_test.csv', 'xor_test.csv']
# testing_network(datasets)


# ------------- training the mnist dataset -------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255
X_test = X_test.reshape(-1, 28*28) / 255

def one_hot(y, n_classes=10):
    oh = np.zeros((y.size, n_classes))
    oh[np.arange(y.size), y] = 1
    return oh

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)
training_data = list(zip(X_train, y_train_oh))
test_data = list(zip(X_test, y_test_oh))

net_mnist =  Network([784, 70, 70, 10], cost='cross entropy')
net_mnist.SGD(training_data, epochs=20, batch_size=25, lr=0.3)
accuracy, prediction = net_mnist.evaluate(test_data)
print('MNIST model Accuracy')
print(accuracy)