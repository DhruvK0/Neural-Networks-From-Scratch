from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt


#load sample images
china = load_sample_image('china.jpg') / 255
flower = load_sample_image('flower.jpg') / 255

images = np.array([china, flower])
batch_size, height, width, channels = images.shape
#max pooling layer
max_pool = keras.layers.MaxPool2D(pool_size=2)

#max pooling layer along depth of feature maps
depth_max_pool = tf.nn.max_pool(images, 
                                ksize=(1,1,1,3),
                                strides=(1,1,1,3),
                                padding="same")

#function included in a keras model
depth_keras_pool = keras.layers.Lambda(lambda X: tf.nn.max_pool(X, ksize=(1,1,1,3), strides=(1,1,1,3)))

