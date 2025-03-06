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
from timeit import default_timer as timer
import numpy as np
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
import itertools
import plotting
import storage
matplotlib.use('TkAgg')
os.environ['KERAS_BACKEND'] = 'tensorflow'

listmap = lambda func, collection: list(map(func, collection))

unzip = lambda a:list(map(list,list(zip(*a))))
pd_cols = ['vals', 'evals', 'pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim', 'full_desc', 'elapsed_time', 'coco_directory', 'timestamp', 'true_eval_budget','train_num', 'sort_train','scale_train'] # and 'ranks', '
func_names = ['Sphere','Ellipsoidal','Rastrigin','B ̈uche-Rastrigin','Linear Slope','Attractive Sector','Step Ellipsoidal','Rosenbrock','Rosenbrock rotated','Ellipsoidal','Discus','Bent Cigar','Sharp Ridge','Different Powers','Rastrigin','Weierstrass','Schaffers F7','Schaffers F7 moderately ill-conditioned','Composite Griewank-Rosenbrock F8F2','Schwefel','Gallagher’s Gaussian 101-me Peaks','Gallagher’s Gaussian 21-hi Peaks','Katsuura','Lunacek bi-Rastrigin']

def get_full_desc1(item):
    nonempty_append = lambda name: ('_'+item[name] if len(item[name])>0 else '')
    names = ['model_name', 'dim_red_name','train_num', 'sort_train','scale_train']
    strs = map(nonempty_append,names)
    full_desc = item['pop_size'] + '_' + item['evo_mode'] +  (''.join(strs))
    return full_desc

def get_full_desc2(pop_size,evo_mode,model_name, dim_red_name,train_num, sort_train,scale_train):
    nonempty_append = lambda val: (f'_{val}' if len(val)>0 else '')
    vals = [model_name, dim_red_name,train_num, sort_train,scale_train]
    strs = map(nonempty_append,vals)
    full_desc = f'{pop_size}_{evo_mode}'+  (''.join(strs))
    return full_desc

def main(df=None):
    if df is None:
        df = storage.load_data()
    df = run(df)
    plot(df)

def plot(df=None):
    if df is None:
        df = storage.load_data()
    plotting.plot(df)


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

    get_nth = lambda n, list: listmap(lambda a: a[n], list)

    def ansamble(combination_f, models): 
        return (
            lambda h,x,y,w,old_model: lambda data: combination_f(np.stack([m(h,x,y,w,None)(data) for m in get_nth(0,models)],0), axis=0), 
            'ansamble_[' + '_'.join((get_nth(1,models))) + ']'
        )
    ans1 =ansamble(np.median,[gp,elm(200), rbf([1/2], 1)])
    configs = [ 
        ## pop_size, evolution_eval_mode, dim_reduction, model, train_num, sort_train, scale_train

        # [1,pure,None,None, -1, False, False],
        [2,pure,None,None, -1, False, False],
        [4,pure,None,None, -1, False, False],
        [8,pure,None,None, -1, False, False],
        [16,pure,None,None, -1, False, False],
        [32,pure,None,None, -1, False, False],
        
        [4,best_k(1.0/2),None,gp, 200, False, False],
        [4,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [4,best_k(1.0/2),vae([1/2]),gp, 200, False, False],
        
        [8,best_k(1.0/2),None,gp, 200, False, False],
        [8,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [8,best_k(1.0/2),vae([1/2]),gp, 200, False, False],

        [16*1,best_k(1.0/2),None,gp, 200, False, False],
        [16*1,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [16*1,best_k(1.0/2),vae([1/2]),gp, 200, False, False],
        
        
        [16*2,best_k(1.0/2),None,gp, 200, False, False],
        [16*2,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [16*2,best_k(1.0/2),vae([1/2]),gp, 200, False, False],

        [16*3,best_k(1.0/2),None,gp, 200, False, False],
        [16*3,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        [16*3,best_k(1.0/2),vae([1/2]),gp, 200, False, False],
        
        [8,best_k(1.0/4),None,gp, 200, False, False],
        [8,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [8,best_k(1.0/4),vae([1/2]),gp, 200, False, False],

        [16*1,best_k(1.0/4),None,gp, 200, False, False],
        [16*1,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [16*1,best_k(1.0/4),vae([1/2]),gp, 200, False, False],
        
        
        [16*2,best_k(1.0/4),None,gp, 200, False, False],
        [16*2,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [16*2,best_k(1.0/4),vae([1/2]),gp, 200, False, False],

        [16*3,best_k(1.0/4),None,gp, 200, False, False],
        [16*3,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        [16*3,best_k(1.0/4),vae([1/2]),gp, 200, False, False],

        [16*1,best_k(1.0/8),None,gp, 200, False, False],
        [16*1,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*1,best_k(1.0/8),vae([1/2]),gp, 200, False, False],
        
        [16*2,best_k(1.0/8),None,gp, 200, False, False],
        [16*2,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*2,best_k(1.0/8),vae([1/2]),gp, 200, False, False],

        [16*3,best_k(1.0/8),None,gp, 200, False, False],
        [16*3,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*3,best_k(1.0/8),vae([1/2]),gp, 200, False, False],
        
        [16*4,best_k(1.0/8),None,gp, 200, False, False],
        [16*4,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        [16*4,best_k(1.0/8),vae([1/2]),gp, 200, False, False],


        [16,best_k(1.0/2),None, ans1, 200, False, False],

        
    ]
    for config in configs:
        res = single_config(config,budget,problem_info, df=df) 
        storage.datastore_store_as_number(res)
        df = pd.concat([df, res], ignore_index=True)

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
        
        return df_row

    # results = Parallel(n_jobs=8, prefer="threads")(delayed(p(run_problem,observer,pops,surrs,dim_red,model, desc,trues,minimal_print))(problem) for problem in suite)
    results = []
    for problem in suite:
        fun, dim, ins = problem.id_triple

        # check whether those settings were already run and are stored in the dataframe
        if df is not None:
            names = ['pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim',  'true_eval_budget', 'train_num', 'sort_train','scale_train']
            vals = [pop_size, evo_mode,model_name,dim_red_name, ins, fun, dim, budget, train_num, sort_train, scale_train]
            masks = np.array([(df[n] == v).to_numpy() for n,v in zip(names,vals)]).T
            is_run_duplicate = np.logical_and.reduce(masks,axis=1)
            is_run_duplicate = np.any(is_run_duplicate)
            if is_run_duplicate:
                continue

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
        results.append(df_row)

    # list(zip(*l))
    # for problem in suite:  # this loop will take several minutes or longer
    #     run_problem(problem,observer)
    print(f"....................................................................run complete {config}")
   
    df = pd.DataFrame({k:v for (k,v) in zip (pd_cols,unzip(results))}) # converts list of dataframe slices to dataframe
    return df



if __name__ == '__main__':
    # ['pure', 'gp', 'elm', rbf]
    df = storage.load_old(None, 'data/data.h5')
    storage.store_data(df, 'df', 'data/datas.h5')
    main()
    # df = run(df)
    # plot(df)
    # datastore_store(load_data(),'w')
    