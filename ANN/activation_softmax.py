from this import d
import numpy as np

class activation_Softmax:
    
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probs = exp_vals/ np.sum(exp_vals, axis=1, keepdims=True)

        self.output = probs

    def backward(self, dvalues):

        #create empty uninitialized array
        self.dinputs = np.empty_like(dvalues)

        #enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            #Flatten output array 
            single_output = single_output.reshape(-1,1)
            
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
