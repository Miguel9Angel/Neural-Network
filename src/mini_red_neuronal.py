import numpy as np
import math

data = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
layers = [2,4,5,1]
weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
biases = [np.random.rand(1, y) for y in layers[1:]]

def sigmoid(z):
    return 1 / (1+np.exp(z))

def output_error(a, y):
    dev_sig = a*(1-a)
    dev_cost = a - y
    return dev_cost * dev_sig

X = data[:, :2]
y = data[:, 2:]
activations = []
a = X
lr = 0.25
for i in range(0, len(layers)-1):
    z = np.dot(a, weights[i])+biases[i]
    a = sigmoid(z)
    activations.append(a)

for i in range(len(activations)):
    print('activation: ', i)
    print(activations[i])
    print('\n')

n_layers = len(layers)-1
errors = np.zeros((1, n_layers))
errors[0, n_layers] = output_error(a, y)
print('Output error: \n', errors)

for idx in range(len(activations)-1, 0):
    error = np.dot(np.transpose(weights[idx]), )+biases[i]