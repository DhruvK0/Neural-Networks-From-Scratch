import numpy as np

class Accuracy():
    # Calculate accuracy from output of activation and targets
    # calculate values along first axis
    def calculate(self, output, y_true):
        self.predictions = np.argmax(output, axis=1)
        if len(y_true.shape) == 2:
            self.y = np.argmax(y_true, axis=1)
        
        return np.mean(self.predictions == self.y)