from VAE import *

vaes = []
# different layer sizes
vaes.append(VAE([784,256,2]))
vaes.append(VAE([698,98,565,45,1]))
vaes.append(VAE([34,2]))

# different decoder activation function
vaes.append(VAE([784,256,2], decoder_activation='relu'))
vaes.append(VAE([784,256,2], decoder_activation='softplus'))
vaes.append(VAE([784,256,2], decoder_activation='sigmoid'))

# different optimizer
vaes.append(VAE([782,256,2], optimizer='adam'))
vaes.append(VAE([782,256,2], optimizer='rmsprop'))


import numpy as np
from keras.datasets import mnist

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

for vae in vaes:
    vae=VAE([784,256,2], verbose=True)
    vae.recognize(x_train, epochs=10, verbose=True)

