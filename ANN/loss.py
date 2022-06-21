import numpy as np

class Loss:
    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers


    def calculate(self, output, y):
        #Calculate Sample Losses
        sample_losses = self.forward(output, y )

        #Calculating mean loss of sample
        data_loss = np.mean(sample_losses)

        return data_loss, self.regularization_loss()
    
    #regularization loss calculation
    def regularization_loss(self, layer):
        # default value of 0
        regularization_loss = 0

        for layer in self.trainable_layers:
            #L1 regularization for weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            #L2 regulariztion for weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            #L1 regularization for biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            #L2 regulariztion for biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        
        return regularization_loss



