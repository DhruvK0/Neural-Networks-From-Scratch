import numpy as np

class activation_Sigmoid():
    #forward pass
    def forward(self, inputs):
        #calculate sigmoid of inputs
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output