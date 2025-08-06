import numpy as np
data = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
layers = [2,4,5,1]
weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
biases = [np.random.rand(1, y) for y in layers[1:]]

X = data[:, :2]
y = data[:, 2:]
activations = []
a = X
for i in range(1, len(layers)):
    z = weights[i-1]