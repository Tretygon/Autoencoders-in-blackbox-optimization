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
from joblib import Parallel, delayed
from cocoex import default_observers
import shutil 

os.environ['KERAS_BACKEND'] = 'tensorflow'
def main():
    # shutil.rmtree("ppdata")
    # shutil.rmtree("exdata")
    ### input
    suite_name = "bbob"
    import datetime
    now = datetime.datetime.now()
    output_folder = "optimize" + str(now.minute) + str(now.second)

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")
    
    # observer_name = another observer if so desired
    
    minimal_print = cocoex.utilities.MiniPrint()
    trues = 400
    all_observers = []
    for pops,surrs,dim_red,model, desc in [
            # [5,Pure(),None,None,'pure5'],
            # [10,Pure(),None,None,'pure10'],
            # [15,Pure(),None,None,'pure15'],
            # [20,Pure(),None,None,'pure20'],
            # [50,Best_k(10,1,1),p(models.vae,[1/2]),p(models.gp,GPK.Matern(nu=5/2)),'vae+gp'],
            [50,Best_k(10,1,1),p(models.pca,1/2),p(models.gp,GPK.Matern(nu=5/2)),'pca+gp'],
            [50,Best_k(5,1,1),p(models.vae,[1/2]),p(models.elm,200),'vae+elm200'],
            [50,Best_k(5,1,1),p(models.pca,1/2),p(models.elm,200),'pca+elm200'],
            # [50,Best_k(10,1,1),models.id,p(models.elm,200),'elm200'],
            # [50,Best_k(10,1,1),models.id,p(models.elm,200),'elm200'],
            [50,Best_k(5,1,1),p(models.pca,1/2),p(models.rbf_network,[1/2],5/2,),'pca+rbf'],
            [50,Best_k(5,1,1),p(models.vae,[1/2]),p(models.rbf_network,[1/2],5/2,),'vae+rbf'],
            # [50,Best_k(5,1,1),models.id,p(models.mlp,[1/4]),'mlp'],
        ]:
        opts = {
            'algorithm_name': desc,
            'algorithm_info': '"ieaieaiea"',
            "result_folder": desc
        }
        observer = cocoex.Observer(suite_name, opts)
        all_observers.append(observer)
        all_ran_funcs = set()
        # results = Parallel(n_jobs=8, prefer="threads")(delayed(p(run_problem,observer,pops,surrs,dim_red,model, desc,trues,minimal_print))(problem) for problem in suite)
        results = [p(run_problem,observer,pops,surrs,dim_red,model, desc,trues,minimal_print)(problem) for problem in suite]
        # for problem in suite:  # this loop will take several minutes or longer
        #     run_problem(problem,observer)
    ### post-process data
    cocopp.genericsettings.background = {}
    cocopp.main(' '.join([o.result_folder for o in all_observers]))
    # cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

def run_problem(observer,pops,surrs,dim_red,model, desc,trues,printer,problem):
    d=problem.dimension
    _,problem_f,ins,_ = problem.id.split('_')
    if not( d==40 and int(problem_f[1:]) <2): return
    # if not((ins == 'i01' or ins == 'i02') and d==40 and int(problem_f[1:]) <5): return
    # all_ran_funcs.add(problem_f)
    observer.observe(problem)
    print(f'---------------------------------------------------f: {problem_f}  dim: {problem.dimension}')
    
    best,best_last = evo.run_surrogate(
                problem,
                pop_size = pops, 
                true_evals=trues, 
                surrogate_usage=surrs,
                dim_red_f = dim_red,
                model_f = model,
                printing=False
            )
    res = f'{pops}, {trues}, {surrs}, {desc}, {round(best,2)}, {round(best_last,2)}'
    print(res)
    problem.free()
    return 42
    printer(problem, final=problem_f=='f24')
if __name__ == '__main__':
    main()