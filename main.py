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
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
import evo
from evo import Alternate_full_generations,Best_k,Pure
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import sklearn.gaussian_process.kernels as GPK
from datetime import datetime
from functools import partial as p
import matplotlib
import models
import plotting
import storage
matplotlib.use('TkAgg')
os.environ['KERAS_BACKEND'] = 'tensorflow'

listmap = lambda func, collection: list(map(func, collection))

unzip = lambda a:list(map(list,list(zip(*a))))
pd_cols = ['vals', 'evals', 'pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim', 'full_desc', 'elapsed_time', 'coco_directory', 'timestamp', 'true_eval_budget','train_num', 'sort_train','scale_train'] # and 'ranks', '
func_names = ['Sphere','Ellipsoidal','Rastrigin','B ̈uche-Rastrigin','Linear Slope','Attractive Sector','Step Ellipsoidal','Rosenbrock','Rosenbrock rotated','Ellipsoidal','Discus','Bent Cigar','Sharp Ridge','Different Powers','Rastrigin','Weierstrass','Schaffers F7','Schaffers F7 moderately ill-conditioned','Composite Griewank-Rosenbrock F8F2','Schwefel','Gallagher’s Gaussian 101-me Peaks','Gallagher’s Gaussian 21-hi Peaks','Katsuura','Lunacek bi-Rastrigin']

def get_full_desc(item):
    nonempty_append = lambda name: ('_'+str(item[name]) if len(str(item[name]))>0 else '')
    names = ['evo_mode','model_name', 'dim_red_name','train_num', 'sort_train','scale_train']
    strs = map(nonempty_append,names)
    full_desc = item['pop_size'] + (''.join(strs))
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
    rbf = lambda layers,gamma: (p(models.rbf_network,layers,gamma), f'rbf{layers}_{gamma}')
    gp = p(models.gp,GPK.Matern(nu=5/2)) , 'gp'
    mlp = lambda nodes: (p(models.mlp,nodes), f'mlp{nodes}')

    
    
    ans1 =models.ansamble.create(np.median,[gp, elm(200), rbf([1/2], 1)])
    configs = [ 
        ## pop_size, evolution_eval_mode, dim_reduction, model, train_num, sort_train, scale_train

        [2,pure,None,None, -1, False, True],
        [4,pure,None,None, -1, False, True],
        [6,pure,None,None, -1, False, True],
        [8,pure,None,None, -1, False, True],
        [16,pure,None,None, -1, False, True],
        [32,pure,None,None, -1, False, True],
        
        # [4,best_k(1.0/2),None,gp, 200, False, False],
        # [4,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        # [4,best_k(1.0/2),vae([1/2]),gp, 200, False, False],
        
        # [8,best_k(1.0/2),None,gp, 200, False, False],
        # [8,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        # [8,best_k(1.0/2),vae([1/2]),gp, 200, False, False],

        # [16*1,best_k(1.0/2),None,gp, 200, False, False],
        # [16*1,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        # [16*1,best_k(1.0/2),vae([1/2]),gp, 200, False, False],
        
        
        # [16*2,best_k(1.0/2),None,gp, 200, False, False],
        # [16*2,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        # [16*2,best_k(1.0/2),vae([1/2]),gp, 200, False, False],

        # [16*3,best_k(1.0/2),None,gp, 200, False, False],
        # [16*3,best_k(1.0/2),pca(1/2),gp, 200, False, False],
        # [16*3,best_k(1.0/2),vae([1/2]),gp, 200, False, False],
        
        # [8,best_k(1.0/4),None,gp, 200, False, False],
        # [8,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        # [8,best_k(1.0/4),vae([1/2]),gp, 200, False, False],

        # [16*1,best_k(1.0/4),None,gp, 200, False, False],
        # [16*1,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        # [16*1,best_k(1.0/4),vae([1/2]),gp, 200, False, False],
        
        
        # [16*2,best_k(1.0/4),None,gp, 200, False, False],
        # [16*2,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        # [16*2,best_k(1.0/4),vae([1/2]),gp, 200, False, False],

        # [16*3,best_k(1.0/4),None,gp, 200, False, False],
        # [16*3,best_k(1.0/4),pca(1/2),gp, 200, False, False],
        # [16*3,best_k(1.0/4),vae([1/2]),gp, 200, False, False],

        # [16*1,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        # [16*1,best_k(1.0/8),vae([1/2]),gp, 200, False, False],
        
        # [16*2,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        # [16*2,best_k(1.0/8),vae([1/2]),gp, 200, False, False],

        # [16*3,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        # [16*3,best_k(1.0/8),vae([1/2]),gp, 200, False, False],
        
        # [16*4,best_k(1.0/8),pca(1/2),gp, 200, False, False],
        # [16*4,best_k(1.0/8),vae([1/2]),gp, 200, False, False],


        # [16,best_k(1.0/2),None, ans1, 200, False, False],

        # [24,best_k(1.0/12),pca(4/10),gp, 200, False, False],
        # [24,best_k(1.0/12),pca(6/10),gp, 200, False, False],
        # [24,best_k(1.0/12),vae([1/2]),gp, 200, False, False],
        # [36,best_k(1.0/12),pca(4/10),gp, 200, False, False],
        # [36,best_k(1.0/12),pca(6/10),gp, 200, False, False],
        # [36,best_k(1.0/12),vae([1/2]),gp, 200, False, False],
        
        # [48,best_k(1.0/16),None,gp, 200, False, False],
        # [48,best_k(1.0/16),pca(1/2),gp, 200, False, False],
        # [48,best_k(1.0/16),vae([1/2]),gp, 200, False, False], 
        # [32,best_k(1.0/16),None,gp, 200, False, False],
        # [32,best_k(1.0/16),pca(1/2),gp, 200, False, False],
        [32,best_k(1.0/16),vae([1/2]),gp, 200, False, False], 


        [64,best_k(1.0/8),pca(1/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(2/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(3/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(4/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(5/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(6/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(7/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(8/10),gp, 200, False, False],
        [64,best_k(1.0/8),pca(9/10),gp, 200, False, False],

        # [24,best_k(1.0/12),pca(5/10),gp, 200, False, False],
        # [36,best_k(1.0/12),pca(5/10),gp, 200, False, False],
    ]
    for model in [rbf([1], 10.0),rbf([5], 5.0),elm(10),elm(30), mlp([100]),mlp([1,1,1]),mlp([10,10,10]), mlp([10,10])]:
        for pop in [48]:
            configs.append([pop,best_k(1.0/8),None,model, 200, False, False])

    for config in configs:
        res = single_config(config,budget,problem_info, df=df) 
        storage.store_data(res, None)
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

    suite = cocoex.Suite("bbob", "", problem_info) 
    minimal_print = cocoex.utilities.MiniPrint()
    observer = cocoex.Observer("bbob", opts)
    df_filtered = df

    # filter already done experiments to find those same as this one
    if df_filtered is not None:
            names = ['model','pop_size', 'evo_mode', 'dim_red', 'true_eval_budget', 'train_num', 'sort_train','scale_train']
            vals = [model_name,pop_size, evo_mode,dim_red_name, budget, train_num, sort_train, scale_train]
            df_filtered = df
            for n,v in zip(names,vals):
                df_filtered = df_filtered[df_filtered[n]==v]
    
    results = []
    for problem in suite:
        fun, dim, ins = problem.id_triple

        # check if this experiment was already run
        if df_filtered is not None:
            names = ['instance','function', 'dim']
            vals = [ins, fun, dim]
            df_filtered_local = df_filtered
            for n,v in zip(names,vals):
                df_filtered_local = df_filtered_local[df_filtered_local[n]==v]
            # masks = np.array([(df[n] == v).to_numpy() for n,v in zip(names,vals)]).T
            # is_run_duplicate = np.logical_and.reduce(masks,axis=1)
            # is_run_duplicate = np.any(is_run_duplicate)
            if not df_filtered_local.empty:#is_run_duplicate:
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
   
    res = pd.DataFrame({k:v for (k,v) in zip (pd_cols,unzip(results))}) # converts list of dataframe slices to dataframe
    return res



if __name__ == '__main__':
    # ['pure', 'gp', 'elm', rbf]
    df = storage.load_data(None, 'data/data.h5')
    # df_og = storage.merge_and_load()
    df = run(df)
    # plot(df)
    # datastore_store(load_data(),'w')
    