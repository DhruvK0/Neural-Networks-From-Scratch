import numpy as np

class activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
