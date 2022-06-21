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

#create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
print(filters)
filters[:, 3, :, 0] = 1   #vertical line
print(filters)
filters[3, :, :, 1] = 1   #horizontal line

outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap="gray")
plt.show()

