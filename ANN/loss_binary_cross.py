import numpy as np

class Loss_BinaryCrossEntropy():
    #forward pass
    def forward(self, y_pred, y_true):
        #preclip values from 1e-7 to 1 - 1e-7
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = (y_true * np.log(y_pred_clipped) + (1- y_true) * np.log(1 - y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

