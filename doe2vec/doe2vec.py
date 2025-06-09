import os.path
import sys
import warnings
from statistics import mode

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
from datasets import load_dataset
from huggingface_hub import from_pretrained_keras
from matplotlib import cm
from mpl_toolkits import mplot3d
from numpy.random import seed
from scipy.stats import qmc
import sklearn
from sklearn import manifold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import tensorflow
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

# from doe2vec import bbobbenchmarks as bbob
from doe2vec.models import VAE, Autoencoder
from doe2vec.modulesRandFunc import generate_exp2fun as genExp2fun
from doe2vec.modulesRandFunc import generate_tree as genTree
from doe2vec.modulesRandFunc import generate_tree2exp as genTree2exp
import numba
def no_descs(ax):
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            for line in axis.get_ticklines():
                line.set_visible(False)

class doe_model:
    def __init__(
        self,
        dim,
        m,
        n=250_000,
        latent_dim=20,
        seed_nr=0,
        kl_weight=0.001,
        custom_sample=None,
        use_mlflow=False,
        mlflow_name="Doc2Vec",
        model_type="VAE",
    ):
        """Doe2Vec model to transform Design of Experiments to feature vectors.

        Args:
            dim (int): Number of dimensions of the DOE
            m (int): Power for number of samples used in the Sobol sampler (not used for custom_sample)
            n (int, optional): Number of generated functions to use a training data. Defaults to 1000.
            latent_dim (int, optional): Number of dimensions in the latent space (vector size). Defaults to 16.
            seed_nr (int, optional): Random seed. Defaults to 0.
            kl_weight (float, optional): Defaults to 0.1.
            custom_sample (array, optional): dim-d Array with a custom sample or None to use Sobol sequences. Defaults to None.
            use_mlflow (bool, optional): To use the mlflow backend to log experiments. Defaults to False.
            mlflow_name (str, optional): The name to log the mlflow experiment. Defaults to "Doc2Vec".
            model_type (str, optional): The model to use, either "AE" or "VAE". Defaults to "VAE".
        """
        self.dim = dim
        self.m = m
        self.n = n
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.seed = seed_nr
        self.use_VAE = False
        self.model_type = model_type
        self.fitted = False
        self.loaded = False
        self.autoencoder = None
        self.functions = []
        self.fun_save_path = f'doe_saves/functions.npy'
        self.save_path = f'doe_saves/{self.dim}_{self.m}_{self.latent_dim}'
        if model_type == "VAE":
            self.use_VAE = True
            self.pure_model_type = self.model_type
            self.model_type = self.model_type + str(kl_weight)
        seed(self.seed)
        # generate the DOE using Sobol
        if custom_sample is None:
            self.sampler = qmc.Sobol(d=self.dim, scramble=False, seed=self.seed)
            self.sample = self.sampler.random_base2(m=self.m)
            # self.sample = np.where(self.sample == 0.0, 1e-2, self.sample)
            self.sample = np.clip(self.sample, 0.01, 0.99)
        else:
            self.sample = custom_sample
        self.use_mlflow = use_mlflow
        if use_mlflow:
            mlflow.set_experiment(mlflow_name)
            mlflow.start_run(
                run_name=f"run {self.dim}-{self.m}-{self.latent_dim}-{self.seed}"
            )
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("dim", self.dim)
            mlflow.log_param("kl_weight", self.kl_weight)
            mlflow.log_param("m", self.m)
            mlflow.log_param("latent_dim", self.latent_dim)
            mlflow.log_param("seed", self.seed)
    def __str__(self):
        return f'doe_{self.dim}-{self.m}-{self.latent_dim}'
    

    def load_or_train(self):
        if self.loaded: return

        if os.path.exists(self.fun_save_path):
            self.functions = np.load(self.fun_save_path)
            # self.functions = [eval('lambda array_x: '+f) for f in self.functions]
            assert(len(self.functions)== self.n)
        
        if os.path.exists(self.save_path):
            self.autoencoder = tf.keras.models.load_model(f'{self.save_path}/model')
            self.Y = np.load(f"{self.save_path}/y.npy")
            self.func_mask = np.load(f"{self.save_path}/func_mask.npy")
            self.functions = self.functions[self.func_mask]
        else: 
            self.generate_functions_and_data(self.functions)
            if not os.path.exists(self.fun_save_path):
                np.save(f"{self.fun_save_path}", self.functions)
            # self.functions = [eval('lambda array_x: '+f) for f in self.functions]
            self.compile()
            self.fit(20)
            os.makedirs(self.save_path)
            self.autoencoder.save(f'{self.save_path}/model')
            np.save(f"{self.save_path}/func_mask.npy",self.func_mask)
            np.save(f"{self.save_path}/y.npy", self.Y)
            
        
        self.loaded = True
    
    def generate_functions_and_data(self, provided_functions=[]):
        def fun_gen():
            for f in provided_functions:
                yield f
            while True:
                tree = genTree.generate_tree(6, 16)
                exp = genTree2exp.generate_tree2exp(tree)
                fun = genExp2fun.generate_exp2fun(exp)
                fun = '('+fun + ')[:,0]'
                yield fun

        self.Y = []
        self.functions =[] 
        self.func_mask = []
        

        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        array_x = self.sample
        iters = 0 
        for fun in fun_gen():
            iters_per_succ = iters/max(len(self.Y),1)
            if len(self.func_mask) >= self.n: break
            iters += 1
            try:
                array_y = eval(fun)
                if (#nonrecoverable
                    np.isnan(array_y).any()
                    or np.isinf(array_y).any()
                    or array_y.ndim != 1
                    or np.any(abs(array_y) < 1e-8)
                    or np.any(abs(array_y) > 1e8)):
                        if len(provided_functions)==self.n: 
                            self.func_mask.append(False) 
                        continue 
                if (np.var(array_y) < 1.0):
                    if (np.var(array_y*10) < 1.0):
                        continue
                    else:
                        fun = '10*('+fun+')'

                        
                self.functions.append(fun)
                self.Y.append(array_y)
                self.func_mask.append(True)

            except Exception as inst:
                if len(provided_functions)==self.n:
                    self.func_mask.append(False)
                continue
        warnings.simplefilter("default")
        # assert(len(provided_functions) == 0 or iters==self.n)
        
        self.Y = self.normalise_y(np.array(self.Y))
        self.functions = np.array(self.functions)
        self.func_mask = np.array(self.func_mask)

    def normalise_y(self,y):
        # m = np.mean(y, axis=1)[:,np.newaxis]
        # std = np.std(y,axis=1)[:,np.newaxis]
        # std = np.where(std==0.0, 1, std)
        # y_normed = (y-m)/std
        mn = np.min(y,axis=1)[:,np.newaxis]
        mx = np.max(y,axis=1)[:,np.newaxis]
        y_normed = (y-mn)/(mx-mn)
        return y_normed
    
    def compile(self):
        self.autoencoder = VAE(self.latent_dim, self.Y.shape[1], kl_weight=self.kl_weight)
        self.autoencoder.compile(optimizer="adam")

    def fit(self, epochs=100,batch_size=128, **kwargs):
        """Fit the autoencoder model.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 100.
            **kwargs (dict, optional): optional arguments for the fit procedure.
        """
        if self.autoencoder is None:
            raise AttributeError("Autoencoder model is not compiled yet")

        self.autoencoder.fit(
                tf.cast(self.Y[:-50], tf.float32),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=((te:=tf.cast(self.Y[-50:], tf.float32)),te),
                **kwargs,
            )

    # finds the closest training function to the one provided
    def func_approx(self, y, scale_inp=True):
        if len(y.shape==1): 
            y = y.reshape(1, -1)
        latent = self.encode(y,norm=True)
        if not self.fitted:
            encoded_Y = self.encode(self.Y)
            self.nn= NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(encoded_Y)
        if len(latent.shape==1): 
            latent = latent.reshape(1, -1)
        dist, i = self.nn.kneighbors(latent)
        i = i[0]
        
        best_approx_str = self.functions[i][0]
        best_approx_f = eval('lambda array_x:'+best_approx_str)

        # real_mean = y.mean()
        # real_std = y.std()
        # approx_y = self.Y[i,:]
        # approx_y_mean = approx_y.mean()
        # approx_y_std = approx_y.std()

        def run_approx(array_x):
            if (added_dim := len(array_x.shape)==1):
                array_x = array_x[np.newaxis,:]
            if scale_inp:
                array_x = (array_x + 5.0)/10 #scale from bbob vals to 0-1
            e = best_approx_f(array_x)
            # if scale:
            #     e = (e-approx_y_mean)/approx_y_std # approx_y 'knows' how is this func scaled, the evaluation needs to be normalised accordingly
            #     e = e *real_std  + real_mean # take the normalised vector and turn it the real values of the true bbob func
            return e[0] if added_dim else e
        # func_approx = lambda array_x:  #scale the apmproximation down to zero mean 1var, then scale this to the y values
        return run_approx, dist[0]
    
    def encode(self, y:np.ndarray, norm=False):
        """Encode a Design of Experiments.

        Args:
            y (array): The DOE to encode.

        Returns:
            array: encoded feature vector.
        """
        
        if len(y.shape) == 1:
            y = y.reshape((1,-1))
        if norm:
            y = self.normalise_y(y)

        y = tf.cast(y, tf.float32)
        if self.use_VAE:
            encoded_doe, _, __ = self.autoencoder.encoder(y)
            encoded_doe = np.array(encoded_doe)
            encoded_doe = np.squeeze(encoded_doe)
        else:
            encoded_doe = self.autoencoder.encoder(y).numpy()
        return encoded_doe
    
    
    def summary(self):
        """Get a summary of the autoencoder model"""
        self.autoencoder.encoder.summary()

    def plot_label_clusters_bbob(self):
        encodings = []
        fuction_groups = []
        for f in range(1, 25):
            for i in range(100):
                fun, opt = bbob.instantiate(f, i)
                bbob_y = np.asarray(list(map(fun, self.sample)))
                array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                    np.max(bbob_y) - np.min(bbob_y)
                )
                encoded = self.encode([array_x])
                encodings.append(encoded[0])
                fuction_groups.append(f)

        X = np.array(encodings)
        y = np.array(fuction_groups).flatten()
        mds = manifold.MDS(
            n_components=2,
            random_state=self.seed,
        )
        embedding = mds.fit_transform(X).T
        # display a 2D plot of the bbob functions in the latent space

        plt.figure(figsize=(12, 10))
        plt.scatter(embedding[0], embedding[1], c=y, cmap=cm.jet)
        plt.colorbar()
        plt.xlabel("")
        plt.ylabel("")

        if self.use_mlflow:
            plt.savefig("latent_space.png")
            mlflow.log_artifact("latent_space.png", "img")
        else:
            plt.savefig(
                f"latent_space_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}-{self.model_type}.png"
            )

    def visualizeTestData(self, n=5):
        """Get a visualisation of the validation data.

        Args:
            n (int, optional): The number of validation DOEs to show. Defaults to 5.
        """
        if self.use_VAE:
            encoded_does, _z_log_var, _z = self.autoencoder.encoder(self.test_data)
        else:
            encoded_does = self.autoencoder.encoder(self.test_data).numpy()
        decoded_does = self.autoencoder.decoder(encoded_does).numpy()
        fig = plt.figure(figsize=(n * 4, 8))
        for i in range(n):
            # display original
            ax = fig.add_subplot(2, n, i + 1, projection="3d")
            ax.plot_trisurf(
                self.sample[:, 0],
                self.sample[:, 1],
                self.test_data[i],
                cmap=cm.jet,
                antialiased=True,
            )
            no_descs(ax)
            plt.title("original")
            plt.gray()

            # display reconstruction
            ax = fig.add_subplot(2, n, i + 1 + n, projection="3d")
            ax.plot_trisurf(
                self.sample[:, 0],
                self.sample[:, 1],
                decoded_does[i],
                cmap=cm.jet,
                antialiased=True,
            )
            no_descs(ax)
            plt.title("reconstructed")
            plt.gray()
        if self.use_mlflow:
            plt.savefig("reconstruction.png")
            mlflow.log_artifact("reconstruction.png", "img")
        else:
            plt.show()


if __name__ == "__main__":
    print()
    # import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # obj = doe_model(
    #     20,
    #     8,
    #     n=50000,
    #     latent_dim=40,
    #     kl_weight=0.001,
    #     use_mlflow=False,
    #     model_type="VAE",
    # )
    # obj.load_from_huggingface()
    # # test the model
    # obj.plot_label_clusters_bbob()
