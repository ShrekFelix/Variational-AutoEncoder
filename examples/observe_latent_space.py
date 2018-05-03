'''
This example demonstrates using VAE.predict() method to project input onto latent space given a trained VAE encoder network.
The VAE is trained with latent dimension equal to 2 so that it could be shown with a 2D picture. 
The more separate the clusters, the better the VAE recognize the test data. 

'''



from VAE import *

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist



# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae=VAE([784,256,2], verbose=True)
vae.recognize(x_train, epochs=50)

x_test_encoded = vae.predict(x_test)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()


