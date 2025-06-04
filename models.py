
import sys
import VAE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import tensorflow_addons as tfa
import sklearn.gaussian_process.kernels as GPK
import progress_bar
import math
from rbf_layer import InitCentersRandom, RBFLayer
from functools import partial as p
from functools import partial
import VAE
from rbf_layer import RBFLayer
import scipy


import GPy



#dimensionality reductions
def id(x,y=None,w=None,model=None): 
    return lambda a:a

def pca(bottleneck,x,w,model):
    inp_dim = x.shape[-1]
    pca_dim = int(bottleneck*inp_dim) if isinstance(bottleneck,float) else bottleneck
    pca_dim = min(pca_dim,x.shape[0])
    pca = PCA(pca_dim).fit(x)
    return pca.transform

def vae(l,x,w,model):
    d = x.shape[-1]
    if model == None:
        model= VAE.VAE(d,l)
        model.compile()
    model.fit(x,x,batch_size = int(x.shape[0]/20),epochs=3,verbose=0)
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

def gp2(kernel,x,y,w,model):
    k = GPy.kern.Matern52(input_dim=x.shape[-1])
    m = GPy.models.GPRegression(x,y,k)
    m.optimize(max_f_eval = 1000)
    m.opti
    # gp.fit(x, y)
    return m.predict
    
def elm(h,x,y,w,model):
    inp_size = x.shape[-1]
    hidden_size = int(h*inp_size)
    input_weights = tf.random.normal([inp_size,hidden_size])
    biases = tf.random.normal([hidden_size])
    h = lambda a: tf.nn.silu(tf.tensordot(a,input_weights,1) + biases)
    # h = lambda a: tf.nn.relu(tf.tensordot(a,input_weights,1) + biases)
    output_weights = tf.tensordot(tf.linalg.pinv(h(tf.cast(x,tf.float32))), tf.cast(y,tf.float32),1)
    inp = tf.keras.layers.Input(shape=inp_size)
    outp = tf.tensordot(h(inp),output_weights,1)
    model = tf.keras.Model(inputs=inp,outputs=outp)
    return model

def rbf_network(layers,gamma,x,y,w,model): 
    d = x.shape[-1]
    layers = [int(d*n) for n in layers]
    if model == None or model.layers[0].input_shape[0][-1] != d:
        inp = tf.keras.layers.Input(shape=d)
        feed = inp
        for n in layers:
            feed = RBFLayer(n,InitCentersRandom(x), gamma)(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            # feed = tf.nn.relu(feed)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        outp = tf.keras.layers.Dense(1)(feed)
        outp = tf.squeeze(outp,-1)
        model = tf.keras.Model(inputs=inp,outputs=outp)

        model.compile(optimizer=tfa.optimizers.AdamW(1e-3),loss = 'mse')
    model.fit(x,y,batch_size = int(x.shape[0]/20),epochs=3,verbose=0)
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
        outp = tf.squeeze(outp,-1)
        model = tf.keras.Model(inputs=inp,outputs=outp)
        model.compile(optimizer=tfa.optimizers.AdamW(1e-3),loss = 'mse')
    model.fit(x,y,batch_size = int(x.shape[0]/10),epochs=5,verbose=0)
    return model






class Ansamble :
    @staticmethod 
    def create(combination_f, models):
        listmap = lambda func, collection: list(map(func, collection))
        self = Ansamble()
        self.combination_f = combination_f
        self.model_fs = listmap(lambda a: a[0], models)
        self.model_descs = listmap(lambda a: a[1], models)
        self.old_models = listmap(lambda _: None, self.model_fs)
        return (
            self,
            'ansamble_[' + '&'.join(self.model_descs) + ']'
        )
    def __call__(self, data):
            called = [m(data) for m in self.models]
            stacked = np.stack(called,0),
            combined = self.combination_f(stacked, axis=0)
            return combined 
    def train(self,x,y):
        self.models = [f(h,x,y,w,m_old) for (f,m_old) in zip(self.model_fs,self.models)]




class Surrogate:
    def __init__(self,model_f,dim_red_f,train_records,sort_training, scale_training):
        self.model_f = model_f if model_f != None else lambda b,c,d,e: lambda a: a
        self.dim_red_f = dim_red_f if dim_red_f != None else lambda b,c,d: lambda a: a
        self.train_records = train_records
        self.sort_training = sort_training
        self.scale_training = scale_training
        self.model = None
        self.dim_red = None
    def __call__(self,x):
            latent = self.dim_red(x)
            y = self.model(latent)
            if self.scale_training:
                y = scaler.inverse_transform(np.expand_dims(y, -1))
                y = np.squeeze(y,-1)
            y = np.nan_to_num(y, nan=4.9)
            return y
    
    def train(self, train_x, train_y):
        
        X = np.array(train_x)
        Y = np.array(train_y) 
    
        xx,yy = X, Y
        # if scale_training:
        #     scaler = StandardScaler()
        #     yy = scaler.fit_transform(np.expand_dims(yy, -1))
        #     yy = np.squeeze(yy,-1)
        # if sort_training_data: 
        #     sorted = np.argsort(yy)
        #     xx, yy = xx[sorted][::-1], yy[sorted][::-1]
        if self.train_records is not None and self.train_records < X.shape[0]: 
            xx = xx[-self.train_records:]
            yy = yy[-self.train_records:]
        # if sort_training_data and num_reconds_for_model_training != None:  # shuffle the sorted data after the bad ones have been cut
        #     rng = np.random.default_rng(seed=seed)
        #     indexes = range(len(yy))
        #     rng.shuffle(indexes)
        #     xx, yy = xx[indexes], yy[indexes]

        
        # k = 1
        # n = len(Y)
        # weights = 1/(k*n + (np.arange(n)))
        # s = np.argsort(Y)
        # # chosen_i = np.random.choice(np.argsort(Y),size=len(Y),p=weights/np.sum(weights))
        # chosen_i = s[:int(len(Y))]
        # np.random.shuffle(chosen_i)
        # X = X[chosen_i,:]
        # Y = Y[chosen_i]
        weights = None
    
        self.dim_red = self.dim_red_f(xx,weights,self.dim_red)
        latentX = self.dim_red(xx)
        self.model = self.model_f(latentX,yy,weights,self.model)
        
        


if __name__ == '__main__':
    import main
    main.main()