import numpy as np

class Loss_BinaryCrossEntropy():
    #forward pass
    def forward(self, y_pred, y_true):
        #preclip values from 1e-7 to 1 - 1e-7
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = (y_true * np.log(y_pred_clipped) + (1- y_true) * np.log(1 - y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    #backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        #clip data from both sides
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        #calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

        #normalize gradient
        self.dinputs = self.dinputs / samples