#Create Model Class
from layer_input import Layer_Input
class Model:

    def __init___(self):
        #create empty list of network layers
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    #set loss and optimizer
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1):

        self.accuracy.init(y)
        
        # Main training loop
        for epoch in range(1, epochs+1):
            # Temporary
            output = self.forward(X)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)

            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')

    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Iterate the objects
        for i in range(layer_count):
        # If it's the first layer,
        # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        
        # Update loss object with trainable layers
            
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)           


    #forward pass
    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output

    
    def backward(self, output, y):
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)