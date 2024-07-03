#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence produces only rudimentary progress messages and does not provide
batch distribution or timing prints, as `example_experiment2.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

>>> import cma  # doctest:+SKIP
>>> def fmin(fun, x0):
...     return cma.fmin(fun, x0, 2, {'verbose':-9})

"""
from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
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


def main():
    
    ### input
    suite_name = "bbob-largescale"
    output_folder = "optimize-fmin"
    fmin = scipy.optimize.fmin
    budget_multiplier = 1  # increase to 10, 100, ...

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    use_surrogate = False
    training_points = 200
    for problem in suite:  # this loop will take several minutes or longer
        if problem.dimension <100: continue
        print(f'---------------------------------------------------dim: {problem.dimension}')
        
        
        d = problem.dimension
        layers = [d/2]
        
        inp = tf.keras.layers.Input(shape=d)
        feed = inp
        for n in layers:
            feed = tf.keras.layers.Dense(n, activation='relu')(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        feed = tf.keras.layers.Dense(1)(feed)
        mlp = tf.keras.Model(inputs=inp,outputs=feed)
        mlp.compile(optimizer=tfa.optimizers.AdamW(1e-4),loss = 'mse')

       
        vae = VAE.VAE(d,layers)
        vae.compile(optimizer=tfa.optimizers.AdamW(1e-4))
        
        def get_surrogate(train_x, train_y):
            d = problem.dimension
            X = np.array(train_x)
            Y = np.array(train_y)
            dim_red = lambda a:a

            
            k = 1
            n = len(Y)
            weights = 1/(k*n + (np.arange(n)))
            s = np.argsort(Y)
            # chosen_i = np.random.choice(np.argsort(Y),size=len(Y),p=weights/np.sum(weights))
            chosen_i = s[:int(len(Y))]
            np.random.shuffle(chosen_i)
            X = X[chosen_i,:]
            Y = Y[chosen_i]
            # VAE
            # vae.fit(X,X,batch_size = int(Y.shape[0]/5),epochs=5,verbose=0)
            # dim_red = vae

            # PCA
            pca_dim = min(int(d/2),X.shape[0])
            # pca = PCA(pca_dim).fit(X)
            # dim_red = lambda a: pca.transform(a)
            # print(pca.explained_variance_ratio_.sum())


            latentX = dim_red(X)


            # MLP            
            # mlp.fit(latentX,Y,batch_size = int(Y.shape[0]/5),epochs=20,verbose=0)
            # model = mlp


            # ELM
            # inp_size = int(layers[-1])
            # hidden_size = int(2*inp_size)
            # input_weights = tf.random.normal([inp_size,hidden_size])
            # biases = tf.random.normal([hidden_size])
            # h = lambda a: tf.nn.relu(tf.tensordot(a,input_weights,1) + biases)
            # output_weights = tf.tensordot(tf.linalg.pinv(h(tf.cast(latentX,tf.float32))), tf.cast(Y,tf.float32),1)
            # inp = tf.keras.layers.Input(shape=inp_size)
            # outp = tf.tensordot(h(inp),output_weights,1)
            # model = tf.keras.Model(inputs=inp,outputs=outp)
 

            # GP
            # kernel =  GPK.DotProduct() + GPK.WhiteKernel() + GPK.Matern(nu=5/2)
            kernel =  GPK.Matern(nu=5/2)
            gp = GaussianProcessRegressor(kernel,alpha=1e-3).fit(latentX, Y)
            model = gp.predict
            
            
            return lambda a: model(dim_red(a))


        problem.observe_with(observer)  # generates the data for cocopp post-processing
        x0 = problem.initial_solution
        # apply restarts while neither the problem is solved nor the budget is exhausted
        # mn= fmin(problem,x0)
        # global_opt = problem(mn)
        
        for pops,trues in [(10,120),(20,60),(30,40),(40,30),(60,20),(5,240),]:
            for surrs in [[0],[1],[0,2],[2]]:
                res = evo.run_surrogate(
                    problem,
                    pop_size = pops, 
                    true_evals=trues, 
                    surrogate_evals_per_true=surrs,
                    new_model_f = get_surrogate,
                    printing=False
                )
                res = round(res,2)
                print(f'{pops}, {trues}, {surrs}, {res}')
        break
        minimal_print(problem, final=problem.index == len(suite) - 1)
    ### post-process data
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    #webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")


if __name__ == '__main__':
    main()