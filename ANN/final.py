import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


from accuracy import Accuracy
from activation_relu import activation_ReLU

#import activation functions
from softmax_loss_cat_cross import Activation_Softmax_CategoricalCrossEntropy

#import adam optimizer
from optimizer_Adam import Optimizer_Adam

#import layer
from layer_dense import Layer
from layer_dropout import Layer_Dropout
#import regularizer
from loss import Loss

#Create Spiral Data
X, y = spiral_data(samples=100, classes=3)


#create dense layer
dense1 = Layer(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#create ReLU activation function
activation1 = activation_ReLU()
#create second dense layer with 64 inputs and 3 outputs
dense2 = Layer(512, 3)
#create dropout layer
dropout1 = Layer_Dropout(0.1)
#create softmax and loss combined function
loss_activation = Activation_Softmax_CategoricalCrossEntropy()
#create optimizer object
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

#train data in loop with 1000 epochs
for epoch in range(1000):

    #forward pass of data
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    #add regularization
    regularized_loss = loss_activation.regularization_loss(dense1) + loss_activation.regularization_loss(dense2)
    loss = data_loss + regularized_loss
    
    #calculate accuracy of weights and biases
    acc_func = Accuracy()
    accuracy = acc_func.calculate(loss_activation.output, y)

    #print values
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (' +
            f'data_loss: {data_loss:.3f}, ' +
            f'reg_loss: {regularized_loss:.3f}), ' +
            f'lr: {optimizer.current_learning_rate}')
        
    #backward pass of neural net
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    #update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()




