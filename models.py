
import sys
import GP
import VAE
import evo
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import tensorflow_addons as tfa
import sklearn.gaussian_process.kernels as GPK
import progress_bar
import math
from rbf_layer import RBFLayer
from functools import partial as p
from functools import partial
import VAE
from rbf_layer import RBFLayer
import scipy


#dimensionality reductions
def id(x,y,w,model): 
    return lambda a:a

def pca(d,x,y,w,model):
    inp_dim = x.shape[-1]
    d = int(d*inp_dim)
    pca_dim = min(d,x.shape[0])
    pca = PCA(pca_dim).fit(x)
    return pca.transform

def vae(l,x,y,w,model):
    d = x.shape[-1]
    if model == None:
        model= VAE.VAE(d,l)
        model.compile()
    model.fit(x,x,batch_size = int(x.shape[0]/5),epochs=5,verbose=0)
    return model


# predictors

def gp(kernel,x,y,w,model):
    # 'custom' optimizer just to set a different maxiter
    def opt(obj_func,initial_theta,bounds): 
        res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={'maxiter':1000},
            )
        return res.x,res.fun
    gp =  GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5,optimizer=opt)
    gp.fit(x, y)
    return gp.predict
    
def elm(h,x,y,w,model):
    inp_size = x.shape[-1]
    hidden_size = int(h*inp_size)
    input_weights = tf.random.normal([inp_size,hidden_size])
    biases = tf.random.normal([hidden_size])
    h = lambda a: tf.nn.silu(tf.tensordot(a,input_weights,1) + biases)
    output_weights = tf.tensordot(tf.linalg.pinv(h(tf.cast(x,tf.float32))), tf.cast(y,tf.float32),1)
    inp = tf.keras.layers.Input(shape=inp_size)
    outp = tf.tensordot(h(inp),output_weights,1)
    model = tf.keras.Model(inputs=inp,outputs=outp)
    return model

def rbf_network(layers,gamma,x,y,w,model): 
    d = x.shape[-1]
    layers = [int (d*n) for n in layers]
    if model == None or model.layers[0].input_shape[0][-1] != d:
        inp = tf.keras.layers.Input(shape=d)
        feed = inp
        for n in map(int,layers):
            feed = RBFLayer(n)(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            # feed = tf.nn.relu(feed)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        outp = tf.keras.layers.Dense(1)(feed)
        outp = tf.squeeze(outp,-1)
        model = tf.keras.Model(inputs=inp,outputs=outp)

        model.compile(optimizer=tfa.optimizers.AdamW(1e-4),loss = 'mse')
    model.fit(x,y,batch_size = int(x.shape[0]/10),epochs=5,verbose=0)
    return model

def mlp(layers,x,y,w,model):
    d = x.shape[-1]
    layers = [int (d*n) for n in layers]
    if model == None or model.layers[0].input_shape[0][-1] != d:
        inp = tf.keras.layers.Input(shape=d)
        feed = inp
        for n in map(int,layers):
            feed = tf.keras.layers.Dense(n)(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            feed = tf.nn.relu(feed)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        outp = tf.keras.layers.Dense(1)(feed)
        model = tf.keras.Model(inputs=inp,outputs=outp)
        model.compile(optimizer=tfa.optimizers.AdamW(1e-4),loss = 'mse')
    model.fit(x,y,batch_size = int(x.shape[0]/10),epochs=5,verbose=0)
    return model

















if __name__ == '__main__':
    import main
    main.main()