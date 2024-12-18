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
import evo
from evo import Alternate_full_generations,Best_k,Pure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from timeit import default_timer as timer
import pandas as pd
# import tensorflow as tf
from sklearn.decomposition import PCA
# import tensorflow_addons as tfa
import sklearn.gaussian_process.kernels as GPK
import progress_bar
import math
from datetime import datetime
from rbf_layer import RBFLayer
from functools import partial as p
from functools import partial
# import models
from joblib import Parallel, delayed
from cocoex import default_observers
import shutil 
import matplotlib.pyplot as plt
import tkinter 
import matplotlib
import models
import plotting
matplotlib.use('TkAgg')
os.environ['KERAS_BACKEND'] = 'tensorflow'

unzip = lambda a:list(map(list,list(zip(*a))))
pd_cols = ['vals', 'evals', 'pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim', 'full_desc', 'elapsed_time', 'coco_directory', 'timestamp', 'true_eval_budget'] # and 'ranks', '
func_names = ['Sphere','Ellipsoidal','Rastrigin','B ̈uche-Rastrigin','Linear Slope','Attractive Sector','Step Ellipsoidal','Rosenbrock','Rosenbrock rotated','Ellipsoidal','Discus','Bent Cigar','Sharp Ridge','Different Powers','Rastrigin','Weierstrass','Schaffers F7','Schaffers F7 moderately ill-conditioned','Composite Griewank-Rosenbrock F8F2','Schwefel','Gallagher’s Gaussian 101-me Peaks','Gallagher’s Gaussian 21-hi Peaks','Katsuura','Lunacek bi-Rastrigin']

def main():
    df = run()
    plot()
def plot(df=None):
    if df==None:
        df = load_data()
    plotting.plot_ranks(df)

def append(df):
    df['vals'] = df['vals'].map(np.array2string)
    df['evals'] = df['evals'].map(np.array2string)
    df['evo_mode'] = df['evo_mode'].map(str)
    print(df.dtypes)
    print(df)
    df.to_hdf('data.h5','df', mode='a')
    
def load_data(name='df'):
    data_storage = pd.HDFStore('data.h5','r')
    df = data_storage[name] if name in data_storage.keys() else None
    return df

def save_data(df, name='df'):
    raise BaseException()
    data_storage = pd.HDFStore('data.h5','w')
    data_storage['df'] = df

def run():
    budget = int(3*10e1)
    problem_info = f"function_indices:1-24 dimensions:10 instance_indices:1-5"

    pure = Pure()
    best_k = lambda a: Best_k(a,1,1)
   
    vae = lambda layers: (p(models.vae,layers), f'vae{layers}')
    pca = lambda n: (p(models.pca,n), f'pca{n}')
    
    elm = lambda nodes: (p(models.elm,nodes), f'elm{nodes}')
    rbf = lambda layers,gamma: (p(models.rbf_network,layers,gamma), f'elm{layers} {gamma}')
    gp = p(models.gp,GPK.Matern(nu=5/2)) , 'gp'

    configs = [ 
        ## pop_size, evolution_eval_mode, dim_reduction, model
        # [5,Pure(),None,None,'pure5'],
        # [10,Pure(),None,None,'pure10'],
        # [15,Pure(),None,None,'pure15'],
        [8,pure,None,None],
        [16,pure,None,None],
        [32,pure,None,None],
        
        [8,best_k(1.0/2),None,gp],
        [8,best_k(1.0/2),pca(1/2),gp],
        [8,best_k(1.0/2),vae([3/4,1/2]),gp],

        [16*1,best_k(1.0/2),None,gp],
        [16*1,best_k(1.0/2),pca(1/2),gp],
        [16*1,best_k(1.0/2),vae([3/4,1/2]),gp],
        
        
        [16*2,best_k(1.0/2),None,gp],
        [16*2,best_k(1.0/2),pca(1/2),gp],
        [16*2,best_k(1.0/2),vae([3/4,1/2]),gp],
        # [16,best_k(1.0/2),None,elm(200)],
        # [16,best_k(1.0/2),pca(1/2),elm(200)],
        # [16,best_k(1.0/2),vae(1/2),elm(200)],
        # [32,pure,None,None],
        # [16*2,best_k(1.0/2),vae(1/2),gp],
        # [16*22,best_k(1.0/2),pca(1/2),gp],
        # [16*2,best_k(1.0/2),id,rbf([1/2], 5/2)],
    ]
    df_slices = [single_config(config,budget,problem_info) for config in configs]
    df = pd(df_slices).reset_index(drop=True)
    return df
    



def single_config(config,budget,problem_info):
    global pd_cols
    pop_size,evo_mode,dim_red,model = config
    (dim_red_f, dim_red_name) = dim_red if dim_red else (None, '')
    (model_f, model_name) = model if model else (None, '')
    full_desc = f'{pop_size}_{evo_mode}'+ ('_' if len(dim_red_name)>0 else '') + f'{dim_red_name}' + ('_' if len(model_name)>0 else '') + f'{model_name}'
    opts = {
        'algorithm_name': full_desc,
        'algorithm_info': '"ieaieaiea"',
        "result_folder": full_desc
    }

    suite = cocoex.Suite("bbob", "", problem_info) # or eg instance_indices:1-5
    minimal_print = cocoex.utilities.MiniPrint()
    observer = cocoex.Observer("bbob", opts)

    def hehea(problem):
        fun, dim, ins = problem.id_triple
        start_time = timer()
        res = p(run_problem,observer,pop_size,evo_mode,dim_red_f,model_f, full_desc,budget,minimal_print)(problem) 
        end_time = timer()
        elapsed = end_time - start_time
        vals,evals = res
        timestamp = datetime.now().strftime("%m_%d___%H_%M_%S")
        df_row = [evals, vals, pop_size,evo_mode,model_name,dim_red_name, ins, fun, dim, full_desc, elapsed, observer.result_folder, timestamp, budget]
        return df_row

    # results = Parallel(n_jobs=8, prefer="threads")(delayed(p(run_problem,observer,pops,surrs,dim_red,model, desc,trues,minimal_print))(problem) for problem in suite)
    results = []
    for problem in suite:
        res = hehea(problem)
        results.append(res)


    # list(zip(*l))
    # for problem in suite:  # this loop will take several minutes or longer
    #     run_problem(problem,observer)
    print(f"....................................................................run complete {config}")
    unzipped = unzip(results)
    df = pd.DataFrame({k:v for (k,v) in zip (pd_cols,unzipped)})
    append(df)
    return df,observer.result_folder

def run_problem(observer,pops,surrs,dim_red,model, desc,budget,printer,problem):
    global suite1
    problem_num,dim,instance = problem.id_function, problem.dimension,problem.id_instance
    # problem1 = suite1.get_problem_by_function_dimension_instance(problem_num, dim, 1)
    observer.observe(problem)
    print(f'---------------------------------------------------f: {problem_num}  dim: {dim}')
    
    bests = evo.run_surrogate(
                problem,
                problem,
                pop_size = pops, 
                true_evals=int(budget), 
                surrogate_usage=surrs,
                dim_red_f = dim_red,
                model_f = model,
                printing=True
            )
    evals, vals = bests
    res = f'{pops}, {budget}, {surrs}, {desc}, {round(np.min(vals),2)}'
    print(res)
    problem.free()
    return evals, vals
    # printer(problem, final=problem_f=='f24')


if __name__ == '__main__':
    main()