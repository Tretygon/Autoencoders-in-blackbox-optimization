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
from evo import Alternate_full_generations,Best_k,Pure
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
import models

def main():
    
    ### input
    suite_name = "bbob"
    output_folder = "optimize-fmin"
    fmin = scipy.optimize.fmin
    budget_multiplier = 1  # increase to 10, 100, ...

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    full_out = []
    all_ran_funcs = set()
    for problem in suite:  # this loop will take several minutes or longer
        if problem.dimension <39: continue
        d=problem.dimension
        _,problem_f,ins,_ = problem.id.split('_')
        if ins != 'i01' or problem_f in all_ran_funcs: continue
        all_ran_funcs.add(problem_f)
        print(f'---------------------------------------------------f: {problem_f}  dim: {problem.dimension}')
        
        trues = 400
        for pops,surrs,dim_red,model, desc in [
                [5,Pure(),None,None,'pure5'],
                [10,Pure(),None,None,'pure10'],
                [15,Pure(),None,None,'pure15'],
                [20,Pure(),None,None,'pure20'],
                [50,Best_k(5,1,1),models.id,p(models.gp,GPK.Matern(nu=5/2)),'gp50'],
                [50,Best_k(5,1,1),models.id,p(models.elm,200),'elm200'],
                [50,Best_k(10,1,1),models.id,p(models.elm,200),'elm200'],
                [50,Best_k(5,1,1),models.id,p(models.rbf_network,[d/4],2.5,),'rbf'],
                [50,Best_k(5,1,1),models.id,p(models.mlp,[d/4]),'mlp'],
            ]:
                best,best_last = evo.run_surrogate(
                    problem,
                    pop_size = pops, 
                    true_evals=trues, 
                    surrogate_usage=surrs,
                    dim_red_f = dim_red,
                    model_f = model,
                    printing=False
                )
                models.reset_globals()
                res = f'{pops}, {trues}, {surrs}, {desc}, {round(best,2)}, {round(best_last,2)}'
                print(res)
        
        minimal_print(problem, final=problem.index == len(suite) - 1)
    ### post-process data
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    #webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")


if __name__ == '__main__':
    main()