import glob

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time




class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, inp_size,layer_sizes):
        super(CVAE, self).__init__()
        self.latent_dim = int(layer_sizes[-1])
        activation = 'relu'

        en_inp = tf.keras.layers.Input(shape=(inp_size))
        feed = en_inp
        for n in layer_sizes[:-1]:
            feed = tf.keras.layers.Dense(n, activation=activation)(feed) + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
        feed = tf.keras.layers.Dense(self.latent_dim*2)(feed)
        self.encoder = tf.keras.Model(inputs=en_inp,outputs=feed)
        
        de_inp =  input=tf.keras.layers.Input(shape=(self.latent_dim,))
        feed = de_inp
        for n in layer_sizes[:-1:-1]:
            print(int(feed.shape[-1] == n))
            feed = tf.keras.layers.Dense(n, activation=activation)(feed) + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
        feed = tf.keras.layers.Dense(inp_size)(feed)
        self.decoder = tf.keras.Model(inputs=de_inp,outputs=feed)
        

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
  
  
    @tf.function
    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @tf.function
    def __call__(self,x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return z


    @tf.function
    def train_step(self, data):
        x,_ = data
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
            logpz = self.log_normal_pdf(z, 0., 0.)
            logqz_x = self.log_normal_pdf(z, mean, logvar)
            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(x, x_logit)

        return {m.name: m.result() for m in self.metrics}
