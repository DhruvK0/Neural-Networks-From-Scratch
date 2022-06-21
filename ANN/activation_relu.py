import numpy as np

class activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
    
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs