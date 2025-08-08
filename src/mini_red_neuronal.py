import numpy as np
import math

# data = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
data = np.array([[0,0,0]])
layers = [2,4,5,1]
weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
biases = [np.random.rand(1, y) for y in layers[1:]]

def sigmoid(z):
    return 1 / (1+np.exp(z))

def dev_sig(a):
    return a*(1-a)

def output_error(a, y):
    return (a - y) * dev_sig(a)

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

n_layers = len(layers)
errors = [output_error(a, y)]
print('Output error: \n', errors)

for idx in range(2, n_layers):
    error = np.dot(errors[-idx+1], weights[-idx+1].T)*dev_sig(activations[-idx])
    errors.insert(0, error)

print('\nlayer errors')
for error in errors:
    print(error)
n_data = len(X)

print('\nweights and bias before: \n')
for w in weights: print(w)
for b in biases: print(b)

for idx in range(0,n_layers-1):
    weights[idx] = weights[idx] - (lr/n_data)*np.sum(np.dot(errors[idx]*activations[idx].T))
    biases[idx] = biases[idx] - (lr/n_data)*np.sum(errors[idx])
    
print('\nweights and bias after: \n')
for w in weights: print(w)
for b in biases: print(b)