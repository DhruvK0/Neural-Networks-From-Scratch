#Create Model Class
from layer_input import Layer_Input
class Model:

    def __init___(self):
        #create empty list of network layers
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    #set loss and optimizer
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1):
        # Main training loop
        for epoch in range(1, epochs+1):
            # Temporary
            pass
    
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
    
    #forward pass
    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output