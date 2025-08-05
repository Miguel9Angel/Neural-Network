import numpy as np
imput = [[0,0]]
layers = [3,4,5,1]
weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
biases = [np.random.rand(1, y) for y in layers[1:]]
