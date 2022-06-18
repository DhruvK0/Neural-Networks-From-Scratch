import numpy as np 

class Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeroes((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs

    def backward(self, dvalues):
        #Gradient calculation on the parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #Gradient Calculation on values
        self.dinputs  = np.dot(dvalues, self.weights.T)
        
