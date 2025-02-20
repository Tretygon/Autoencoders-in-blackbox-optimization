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
arr2str = lambda arr: ' '.join(str(x) for x in arr)
str2arr = lambda dtype: lambda string: np.fromstring(string, dtype=dtype, sep=' ')
def str2evomode(s):
        if s.startswith("Pure"): return Pure()
        elif s.startswith("BestK"):
            s1 = s[len("BestK"):]
            return Best_k(float(s1) if '.' in s1 else int(s1), 1, 1) 
        else: raise Exception
unzip = lambda a:list(map(list,list(zip(*a))))
pd_cols = ['vals', 'evals', 'pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim', 'full_desc', 'elapsed_time', 'coco_directory', 'timestamp', 'true_eval_budget','train_num', 'sort_train','scale_train'] # and 'ranks', '
func_names = ['Sphere','Ellipsoidal','Rastrigin','B ̈uche-Rastrigin','Linear Slope','Attractive Sector','Step Ellipsoidal','Rosenbrock','Rosenbrock rotated','Ellipsoidal','Discus','Bent Cigar','Sharp Ridge','Different Powers','Rastrigin','Weierstrass','Schaffers F7','Schaffers F7 moderately ill-conditioned','Composite Griewank-Rosenbrock F8F2','Schwefel','Gallagher’s Gaussian 101-me Peaks','Gallagher’s Gaussian 21-hi Peaks','Katsuura','Lunacek bi-Rastrigin']

def main(df=None):
    if df is None:
        df = load_data()
    df = run(df)
    plot(df)

def plot(df=None):
    if df is None:
        df = load_data()
    plotting.plot_ranks(df)

def datastore_store(df):
    df = df.copy()
    df['vals'] = df['vals'].map(arr2str)
    df['evals'] = df['evals'].map(arr2str)
    df['evo_mode'] = df['evo_mode'].map(str)
    df.to_hdf('data.h5','df', mode='a')
    


def load_data(name='df'):
    try:
        with pd.HDFStore('data.h5','r') as data_storage:
            df = data_storage[name]
            df['vals'] = df['vals'].map(str2arr(float))
            df['evals'] = df['evals'].map(str2arr(int))
            df['evo_mode'] = df['evo_mode'].map(str2evomode)
            return df
    except FileNotFoundError:
                return None
    

def save_data(df, name='df'):
    raise BaseException()
    data_storage = pd.HDFStore('data.h5','w')
    data_storage['df'] = df




def run(df=None):
    budget = int(3*10e1)
    problem_info = f"function_indices:1-24 dimensions:10 instance_indices:1-3"

    pure = Pure()
    best_k = lambda a: Best_k(a,1,1)
   
    vae = lambda layers: (p(models.vae,layers), f'vae{layers}')
    pca = lambda n: (p(models.pca,n), f'pca{n}')
    
    elm = lambda nodes: (p(models.elm,nodes), f'elm{nodes}')
    rbf = lambda layers,gamma: (p(models.rbf_network,layers,gamma), f'elm{layers} {gamma}')
    gp = p(models.gp,GPK.Matern(nu=5/2)) , 'gp'



    ansamble = lambda models,combination_f: lambda data: combination_f(np.stack([m(data) for m in models],0), axis=0)

    configs = [ 
        ## pop_size, evolution_eval_mode, dim_reduction, model, train_num, sort_train, scale_train
        [8,pure,None,None, -1, False, False],
        [16,pure,None,None, -1, False, False],
        [32,pure,None,None, -1, False, False],
        
        [8,best_k(1.0/2),None,gp, 200, False, False],
        [8,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [8,best_k(1.0/2),vae([3/4,1/2]),gp, 200, False, False],

        [16*1,best_k(1.0/2),None,gp, 200, False, False],
        [16*1,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [16*1,best_k(1.0/2),vae([3/4,1/2]),gp, 200, False, False],
        
        
        [16*2,best_k(1.0/2),None,gp, 200, False, False],
        [16*2,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [16*2,best_k(1.0/2),vae([3/4,1/2]),gp, 200, False, False],

        [16*3,best_k(1.0/2),None,gp, 200, False, False],
        [16*3,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [16*3,best_k(1.0/2),vae([3/4,1/2]),gp, 200, False, False],
        
        [8,best_k(1.0/4),None,gp, 200, False, False],
        [8,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [8,best_k(1.0/4),vae([3/4,1/2]),gp, 200, False, False],

        [16*1,best_k(1.0/4),None,gp, 200, False, False],
        [16*1,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [16*1,best_k(1.0/4),vae([3/4,1/2]),gp, 200, False, False],
        
        
        [16*2,best_k(1.0/4),None,gp, 200, False, False],
        [16*2,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [16*2,best_k(1.0/4),vae([3/4,1/2]),gp, 200, False, False],

        [16*3,best_k(1.0/4),None,gp, 200, False, False],
        [16*3,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [16*3,best_k(1.0/4),vae([3/4,1/2]),gp, 200, False, False],
        
        [8,best_k(1.0/8),None,gp, 200, False, False],
        [8,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [8,best_k(1.0/8),vae([3/4,1/2]),gp, 200, False, False],

        [16*1,best_k(1.0/8),None,gp, 200, False, False],
        [16*1,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*1,best_k(1.0/8),vae([3/4,1/2]),gp, 200, False, False],
        
        [16*2,best_k(1.0/8),None,gp, 200, False, False],
        [16*2,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*2,best_k(1.0/8),vae([3/4,1/2]),gp, 200, False, False],

        [16*3,best_k(1.0/8),None,gp, 200, False, False],
        [16*3,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*3,best_k(1.0/8),vae([3/4,1/2]),gp, 200, False, False],
        
        [16*4,best_k(1.0/8),None,gp, 200, False, False],
        [16*4,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*4,best_k(1.0/8),vae([3/4,1/2]),gp, 200, False, False],

        
    ]
    for config in configs:
        res = single_config(config,budget,problem_info, df=df) 
        df = pd.concat([df, res], ignore_index=True)
        datastore_store(df)

    return df
    



def single_config(config,budget,problem_info, df=None):
    global pd_cols
    pop_size,evo_mode,dim_red,model,train_num, sort_train,scale_train = config
    pop_size = int(pop_size)
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

    def run_problem(problem):
        fun, dim, ins = problem.id_triple

        # check whether those settings were already run and are stored in the dataframe
        if df is not None:
            names = ['pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim',  'true_eval_budget', 'train_num', 'sort_train','scale_train']
            vals = [pop_size, evo_mode,model_name,dim_red_name, ins, fun, dim, budget, train_num, sort_train, scale_train]
            masks = np.array([(df[n] == v).to_numpy() for n,v in zip(names,vals)]).T
            is_run_duplicate = np.logical_and.reduce(masks,axis=1)
            is_run_duplicate = np.any(is_run_duplicate)
            if is_run_duplicate:
                return None

        observer.observe(problem)
        
        start_time = timer()
        evals, vals = evo.run_surrogate(
            problem,
            problem,
            pop_size = pop_size, 
            true_evals=int(budget), 
            surrogate_usage=evo_mode,
            dim_red_f = dim_red_f,
            model_f = model_f,
            printing=True, 
            seed= 42,
            train_num = train_num, 
            sort_train = sort_train, 
            scale_train= scale_train
        )
        end_time = timer()

        info = f'{pop_size}, {budget}, {evo_mode}, {full_desc}, {round(np.min(vals),2)}'
        problem.free()
        elapsed = end_time - start_time
        timestamp = datetime.now().strftime("%m_%d___%H_%M_%S")
        df_row = [vals, evals, pop_size,evo_mode,model_name,dim_red_name, ins, fun, dim, full_desc, elapsed, observer.result_folder, timestamp, budget, train_num, sort_train,scale_train]
        return df_row

    # results = Parallel(n_jobs=8, prefer="threads")(delayed(p(run_problem,observer,pops,surrs,dim_red,model, desc,trues,minimal_print))(problem) for problem in suite)
    results = []
    for problem in suite:
        res = run_problem(problem)
        if res != None:
            results.append(res)


    # list(zip(*l))
    # for problem in suite:  # this loop will take several minutes or longer
    #     run_problem(problem,observer)
    print(f"....................................................................run complete {config}")
   
    df = pd.DataFrame({k:v for (k,v) in zip (pd_cols,unzip(results))}) # converts list of dataframe slices to dataframe
    return df



if __name__ == '__main__':
    df = load_data()
    # df = run(df)
    df = run(df)
    plot(df)
    # datastore_store(load_data(),'w')
    