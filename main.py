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
import GP
import VAE
import evo
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import tensorflow as tf


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
        if problem.dimension <300: continue
        print(f'---------------------------------------------------dim: {problem.dimension}')
        
        
        d = problem.dimension
        layers = [d/4,d/4]
        
        inp = tf.keras.layers.Input(shape=(d))
        feed = inp
        for n in layers:
            feed = tf.keras.layers.Dense(n, activation='relu')(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        feed = tf.keras.layers.Dense(1)(feed)
        mlp = tf.keras.Model(inputs=inp,outputs=feed)
        mlp.compile(optimizer='adam',loss = 'mse')

       
        model = VAE.CVAE(d,layers)
        model.compile(optimizer='adam')

        def get_surrogate(trainxy):
            nonlocal model
            train_x,train_y = zip(*trainxy)
            d = problem.dimension
            layers = [d/4,d/4]
            X = np.array(train_x)
            Y = np.array(train_y)
            # model.fit(X,X,batch_size = 8,epochs=10,verbose=0,metrics=['mse'])
            

            
            #model.fit(X,Y,batch_size = 8,epochs=10,verbose=1)
            # latent_X = model(X) 
            
            
            latent_X =X 
            # gpr = GaussianProcessRegressor(1**2 * RBF(length_scale=1.0),n_restarts_optimizer=10).fit(latent_X, Y)

            def predict(x):
                # y = model(x)
                # latent = model(x)
                # y = gpr.predict(x)
                r = []
                for a in list(x):
                    y = GP.gaussian_process_predict_mean(X,Y,a)
                    r.append(y)
                y = np.array(r)
                return y
            return predict   


        problem.observe_with(observer)  # generates the data for cocopp post-processing
        x0 = problem.initial_solution
        # apply restarts while neither the problem is solved nor the budget is exhausted
        # mn= fmin(problem,x0)
        # global_opt = problem(mn)
        evo.run_surrogate(
            problem,
            pop_size = 100, 
            generations = 50, 
            new_model_f = get_surrogate,
            max_model_uses = 1, 
            true_eval_every_n = 2
        )

        minimal_print(problem, final=problem.index == len(suite) - 1)
        
    ### post-process data
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    #webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

if __name__ == '__main__':
    main()