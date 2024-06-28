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
from scipy.linalg import pinv
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
    for problem in suite:  # this loop will take several minutes or longer
        if problem.dimension <100: continue
        print(f'---------------------------------------------------dim: {problem.dimension}')
        
        
        d = problem.dimension
        layers = [d/4]
        
        inp = tf.keras.layers.Input(shape=d)
        feed = inp
        for n in layers:
            feed = tf.keras.layers.Dense(n, activation='relu')(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        feed = tf.keras.layers.Dense(1)(feed)
        mlp = tf.keras.Model(inputs=inp,outputs=feed)
        mlp.compile(optimizer=tfa.optimizers.AdamW(1e-4),loss = 'mse')

       
        cvae = VAE.CVAE(d,layers)
        cvae.compile(optimizer=tfa.optimizers.AdamW(1e-4))
        
        def get_surrogate(train_x, train_y):
            d = problem.dimension
            X = np.array(train_x)
            Y = np.array(train_y)
            dim_red = lambda a:a

            # VAE
            # cvae.fit(X,X,batch_size = 4,epochs=10,verbose=0)
            # dim_red = cvae

            # PCA
            # pca = PCA(50).fit(X)
            # dim_red = lambda a: pca.predict(a)
            # print(pca.explained_variance_ratio_.sum())


            latentX = dim_red(X)


            # MLP            
            # mlp.fit(latentX,Y,batch_size = 4,epochs=20,verbose=0)
            # model = mlp


            # ELM
            input_weights = np.random.normal(size=[d,int(2*d)])
            biases = np.random.normal(size=[int(2*d)])
            h = lambda a: np.maximum(np.dot(a, input_weights) + biases,0)
            output_weights = np.dot(pinv(h(latentX)), Y)
            model = lambda a: np.dot(h(a), output_weights)
 

            # GP
            # kernel =  GPK.DotProduct() + GPK.WhiteKernel() 
            # gp = GaussianProcessRegressor(kernel).fit(latent_X, Y)
            # model = gp.predict
            
            
            return lambda a: model(dim_red(a))


        problem.observe_with(observer)  # generates the data for cocopp post-processing
        x0 = problem.initial_solution
        # apply restarts while neither the problem is solved nor the budget is exhausted
        # mn= fmin(problem,x0)
        # global_opt = problem(mn)
        
        for pops,trues in [(5,240),(10,120),(20,60),(30,40),(40,30),(60,20)]:
            for surrs in [[0],[1]]:
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