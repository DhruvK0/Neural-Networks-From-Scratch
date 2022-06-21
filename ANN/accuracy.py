import numpy as np

class Accuracy():
    # Calculate accuracy from output of activation and targets
    # calculate values along first axis
    def calculate(self, predictions, y):
        #get comparison results
        comparisons = self.compare(predictions, y)

        #calculate accuracy
        accuracy = np.mea(comparisons)

        return accuracy