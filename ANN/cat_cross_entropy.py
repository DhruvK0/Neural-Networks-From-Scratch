import numpy as np
from loss import Loss

class Categorical_Cross_Entropy(Loss):
    #loss calculation for forward pass
    def forward(self, y_pred, y_true):
    
        #first get length of sample matrix
        samples = len(y_pred)  

        #clip values
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #if labels are categorical
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        #if labels are one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=1)

        #calculate log probabilities
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods