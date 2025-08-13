import numpy as np
import math
import random
import os

class Network():
    def __init__(self, layers, seed=42, cost='quadratic'):
        np.random.seed(seed)
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        if cost == 'quadratic':
            self.cost = self.quadratuc_cost
        elif cost == 'cross entropy':
            self.cost = self.cross_entropy_cost

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def dev_sig(self, a):
        return a*(1-a)
    
    def quadratic_cost(self, y_true, y_pred, derivate=False):
        if derivate:
            delta = (y_pred - y_true) * self.dev_sig(y_pred)
            return delta
        else:
            m = y_true.shape[0]
            cost = (1/(2*m))*np.sum((y_pred - y_true)**2)
            return cost
    
    def cross_entropy_cost(self, y_true, y_pred, derivate=False):
        if derivate:
            return y_pred-y_true
        else:
            cost = np.sum(np.nan_to_num(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)))
            return cost

    def SGD(self, training_data, epochs=10, batch_size=32, lr=0.1, return_training_cost=False):
        n = len(training_data)
        all_training_costs = []
        for _ in range(epochs):
            random.shuffle( training_data )
            batches = [ training_data [i:i+ batch_size ] for i in range(0, n, batch_size)]
            epoch_cost = []
            for batch in batches:
                X_batch = np.vstack([x for x, _ in batch])
                y_batch = np.vstack([y for _, y in batch])
                activations = self.feedforward(X_batch)
                grads_w, grads_b = self.backpropagation(activations, y_batch, self.cost)
                self.update_w_b(grads_w, grads_b, lr, X_batch.shape[0])
                if return_training_cost:
                    cost = self.cost_function(activations, y_batch)
                    epoch_cost.append(cost)
            if return_training_cost:
                all_training_costs.append(np.mean(epoch_cost) if epoch_cost else 0)
    
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
        
        delta = self.cost(y, activations[-1], derivate=True)
        
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
            
    def cost_function(activations, y):
        y_pred = activations[-1]
        cost = self.cost(y, y_pred, derivate=False)/len(y_pred)
        return cost  
        
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
