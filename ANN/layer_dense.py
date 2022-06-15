import numpy as np 

class Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeroes((1, neurons))

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass
