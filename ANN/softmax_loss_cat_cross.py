import numpy as np
from activation_softmax import activation_Softmax
from cat_cross_entropy import Categorical_Cross_Entropy

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_CategoricalCrossEntropy():
    #construct objects for loss and activation functions
    def __init_(self):
        self.activation = activation_Softmax()
        self.loss = Categorical_Cross_Entropy()

    def forward(self, inputs, y_true):
        #run activation function through previos layer outputs
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        
        #No. of Samples
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples    