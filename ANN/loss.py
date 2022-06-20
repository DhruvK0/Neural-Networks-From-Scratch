import numpy as np

class Loss:
    def calculate(self, output, y):
        #Calculate Sample Losses
        sample_losses = self.forward(output, y )

        #Calculating mean loss of sample
        data_loss = np.mean(sample_losses)

        return data_loss

