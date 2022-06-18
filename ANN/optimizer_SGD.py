import numpy as np

class Optimizer_SGD:
    #initialize otpimizer with learning rate
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    #call once before parameter updates
    def pre_update_params(self):
        #if we use learning rate decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    #update parameters
    def update_params(self, layer):
        #if we use momentum
        if self.momentum:
            #If layer does not have momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                #create momentum for biases
                layer.bias_momentums = np.zeros_like(layer.biases)

            #create weight updates with added momentum - using previous update
            # updates multiplied by retainment factor and updated with current gradient
            weight_updates = self.momentum * layer.weight_momentums - (self.current_learning_rate * self.dweights)
            layer.weight_momentums = weight_updates

            #create bias updates
            bias_updates = self.momentum * layer.bias_momentums - (self.current_learning_rate * self.dbiases)
            layer.bias_momentums = bias_updates

        #vanilla SGD (without momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    #call after parameter updates
    def post_update_params(self):
        self.iterations += 1
