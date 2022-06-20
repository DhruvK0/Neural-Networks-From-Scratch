import numpy as np

class Accuracy():
    # Calculate accuracy from output of activation and targets
    # calculate values along first axis
    def calculate(self, output, y_true):
        self.y_true = y_true
        self.predictions = np.argmax(output, axis=1)
        if len(self.y_true.shape) == 2:
            self.y_true = np.argmax(self.y_true, axis=1)
        
        return np.mean(self.predictions==self.y_true)