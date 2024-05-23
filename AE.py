import glob

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time




class Autoencoder(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, inp_size,layer_sizes):
        super(Autoencoder, self).__init__()
        self.latent_dim = layer_sizes[:-1]
        activation = 'relu'
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(inp_size, 1))]
            +[tf.keras.layers.Dense(n, activation=activation) for n in layer_sizes[:-1]] 
            +[tf.keras.layers.Dense(self.latent_dim)]
        )

        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))]
            +[tf.keras.layers.Dense(n, activation=activation) for n in layer_sizes[:-1:-1]] 
            +[tf.keras.layers.Dense(inp_size)]
        )

    @tf.function
    def __call__(self, x):
        latent = self.encode(x)
        res = self.decode(latent)
        return res

    @tf.function
    def encode(self, x):
        latent = self.encode(x)
        return latent



    @tf.function
    def train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.MeanSquaredError()(x,self(x))
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        