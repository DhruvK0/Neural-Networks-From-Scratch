import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


from accuracy import Accuracy
from activation_relu import activation_ReLU
from activation_sigmoid import activation_Sigmoid
#import activation functions
from softmax_loss_cat_cross import Activation_Softmax_CategoricalCrossEntropy

#import adam optimizer
from optimizer_Adam import Optimizer_Adam

#import layer
from layer_dense import Layer
from layer_dropout import Layer_Dropout
#import regularizer
from loss import Loss
from loss_binary_cross import Loss_BinaryCrossEntropy
#Create Spiral Data
X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1, 1)

#create dense layer
dense1 = Layer(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#create ReLU activation function
activation1 = activation_ReLU()
#create sigmoid activation
activation2 = activation_Sigmoid()
#create second dense layer with 64 inputs and 3 outputs
dense2 = Layer(64, 1)

#create softmax and loss combined function
loss_function = Loss_BinaryCrossEntropy()
#create optimizer object
optimizer = Optimizer_Adam(decay=5e-7)

#train data in loop with 1000 epochs
for epoch in range(1001):

    #forward pass of data
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    data_loss = loss_function.calculate(activation2.output, y)
    #add regularization
    regularized_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularized_loss
    
    #calculate accuracy of weights and biases
    # acc_func = Accuracy()
    # accuracy = acc_func.calculate(loss_function.output, y)

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)
    #print values
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (' +
            f'data_loss: {data_loss:.3f}, ' +
            f'reg_loss: {regularized_loss:.3f}), ' +
            f'lr: {optimizer.current_learning_rate}')
        
    #backward pass of neural net
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()




