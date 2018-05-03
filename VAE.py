#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.metrics import binary_crossentropy

class VAE():
    '''
    Variational Auto Encoder implemented in Keras(using Tensorflow backend) interface
    Original paper can be found on: https://arxiv.org/abs/1312.6114
    
    Returns a compiled VAE model that looks like this:
    
    output            ooooooooooooooooooooooooo
                         XXXXXXXXXXXXXXXXXXXX    -Dense
                         oooooooooooooooooooo
    decoder               **hidden layers**      -Dense
                          ooooooooooooooooo
                                XXXXXX           -Dense
    z                          oooooooo
                               XXX  XXX          -Lambda: sample
    z_mean,z_log_var      oooooooo  oooooooo  
                           XXXXXX    XXXXXX      -Dense
                          ooooooooooooooooo
    encoder               **hidden layers**      -Dense
                        oooooooooooooooooooo
                        XXXXXXXXXXXXXXXXXXXX     -Dense
    input             ooooooooooooooooooooooooo
    
    ----------------------------------------------------------------------------------
    Parameters:
    - layer_sizes:
        Specify layer sizes from input layer to latent space layer.
        The decoding layer is symmetric to the encoding layer.
        At least 1 hidden layer is suggested, so the total length of this parameter is at least 3.
    - decoder_activation:(default 'sigmoid')
        Activation function used in decoding layers.
        As MNIST is real-valued between 0 and 1, sigmoid is proper for it.
    - optimizer:(default 'rmsprop')
        Gradient descent method used in back-propagation.
    - verbose:(default True)
        Will print model summary is set to True.
    '''    
    
    def __init__(self, layer_sizes, decoder_activation='sigmoid', optimizer='rmsprop',verbose=True):
        self.optimizer=optimizer
        
        def sample(args): # used in sampling layer
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], layer_sizes[-1]))
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        layers = [Input(shape=(layer_sizes[0],))]        
        for i in range(1, len(layer_sizes)-1):
            layers.append(Dense(layer_sizes[i], activation='relu')(layers[i-1]))
        
        z_mean = Dense(layer_sizes[-1])(layers[-1])
        z_log_var = Dense(layer_sizes[-1])(layers[-1])
        layers.append(Lambda(sample, output_shape=(layer_sizes[-1],))([z_mean, z_log_var]))
                
        decoders = [] # stores intermediate decoding layer object that can be used to construct generator and the rest of vae model
        decoder_layers = [Input(shape=(layer_sizes[-1],))] # generator takes sample from latent space
        for i in range(len(layer_sizes)-1):
            if i==len(layer_sizes)-2:
                decoders.append(Dense(layer_sizes[-i-2], activation=decoder_activation)) # intermediate decoding layer object
            else:
                decoders.append(Dense(layer_sizes[-i-2], activation='relu')) #
            layers.append(decoders[i](layers[-1])) # construct rest of vae model
            decoder_layers.append(decoders[i](decoder_layers[-1])) # construct generator

        # loss
        recons = layer_sizes[0] * binary_crossentropy(layers[0], layers[-1])
        kl = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        loss = K.mean(recons + kl)
        
        # vae
        self.vae = Model(layers[0], layers[-1])
        self.vae.add_loss(loss)
        self.vae.compile(optimizer=self.optimizer)
        
        if verbose:
            self.vae.summary()

        # model that projects inputs on the latent space
        self.encoder = Model(layers[0], z_mean)
        
        # generator that samples from the learned distribution        
        self.generator = Model(decoder_layers[0], decoder_layers[-1])            
            

    def recognize(self, x_train, batch_size=100, epochs=50, verbose=True):
        self.vae.fit(x_train, epochs=epochs, batch_size=batch_size, verbose = verbose)
       
    def predict(self, x_test, batch_size=100):
        return self.encoder.predict(x_test, batch_size=batch_size)

    def generate(self, sample):
        return self.generator.predict(sample)
