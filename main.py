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
import cocoex.function
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
import ranks
import storage
import pd_cols
from doe2vec.doe2vec import doe_model 
matplotlib.use('TkAgg')
os.environ['KERAS_BACKEND'] = 'tensorflow'
#
# num of sampling points 
# latent size
# knn > 1 as ansamble?
# diff similarities
# 
# sampling -5 5 instead of 0 1
# 
# 
# count the sample eval as true eval or not
# 
# 
# 
# 
# 
# 
dim = 10
budget = int(30*dim) # times dim


listmap = lambda func, collection: list(map(func, collection))

unzip = lambda a:list(map(list,list(zip(*a))))
func_names = ['Sphere','Ellipsoidal','Rastrigin','B ̈uche-Rastrigin','Linear Slope','Attractive Sector','Step Ellipsoidal','Rosenbrock','Rosenbrock rotated','Ellipsoidal','Discus','Bent Cigar','Sharp Ridge','Different Powers','Rastrigin','Weierstrass','Schaffers F7','Schaffers F7 moderately ill-conditioned','Composite Griewank-Rosenbrock F8F2','Schwefel','Gallagher’s Gaussian 101-me Peaks','Gallagher’s Gaussian 21-hi Peaks','Katsuura','Lunacek bi-Rastrigin']


def main(df=None):
    if df is None:
        df = storage.load_data()
    df = run(df)
    plot(df)

def plot(df=None):
    if df is None:
        df = storage.load_data()
    ranks.plot(df)

note = 'no_mean'
def run(df=None):
    global dim, budget
    problem_info = f"function_indices:1-24 dimensions:{dim} instance_indices:1-3"

    pure = Pure()
    best_k = lambda a: Best_k(a)
    doe = lambda latent, power, follow_up,: evo.Doe(follow_up,dim=dim,power=power,latent=latent)
   
    vae = lambda layers: (p(models.vae,layers), f'vae{layers}')
    pca = lambda n: (p(models.pca,n), f'pca{n}')
    
    elm = lambda nodes: (p(models.elm,nodes), f'elm{nodes}')
    rbf = lambda layers,gamma: (p(models.rbf_network,layers,gamma), f'rbf{layers}_{gamma}')
    gp = p(models.gp,GPK.Matern(nu=5/2)) , 'gp'
    mlp = lambda nodes: (p(models.mlp,nodes), f'mlp{nodes}')

    
    
    # ans1 =models.ansamble.create(np.median,[gp, elm(200), rbf([1/2], 1)])

    configs = [ 
        ## pop_size, evolution_eval_mode, dim_reduction, model,budget, train_num, sort_train, scale_train, cma_sees_approximations

        # [2,pure,None,None, budget,-1, False, False, False],
        # [4,pure,None,None, budget,-1, False, False, False],
        # [6,pure,None,None, budget,-1, False, False, False],
        # [8,pure,None,None, budget,-1, False, False, False],
        # [12,pure,None,None,budget, -1, False, False, False],
        # [16,pure,None,None,budget, -1, False, False, False],
        # [24,pure,None,None,budget, -1, False, False, False],
        # [32,pure,None,None,budget, -1, False, False, False],
        



        
        # [48,best_k(1.0/16),None,gp, budget, 200, False, False, False],


        # [64,best_k(1.0/8),pca(1/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(2/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(3/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(4/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(5/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(6/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(7/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(8/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),pca(9/10),gp,budget, 200, False, False, False],
        # [64,best_k(1.0/8),None,gp, budget, 200, False, False, False],
        # [64,best_k(1.0/8),None,gp, budget, 200, False, False, True],

        # [24,best_k(1.0/12),pca(5/10),gp,budget 200, False, False],
        # [36,best_k(1.0/12),pca(5/10),gp, 200, False, False],


        # [8,doe(40, 1, pure),None,None, budget, 200, False, False, False],
        # [8,doe(40, 2, pure),None,None, budget, 200, False, False, False],
        # [8,doe(40, 3, pure),None,None, budget, 200, False, False, False],
        # [8,doe(40, 4, pure),None,None, budget, 200, False, False, False],
        # [8,doe(40, 5, pure),None,None, budget, 200, False, False, False],
        # [8,doe(40, 6, pure),None,None, budget, 200, False, False, False],
        # [8,doe(40, 7, pure),None,None, budget, 200, False, False, False],
        [8,doe(100, 8, pure),None,None, budget, 201, False, False, False],
        # [8,doe(40, 9, pure),None,None, budget, 200, False, False, False],
        # [8,doe(40, 10, pure),None,None, budget, 200, False, False, False],
    ]
    # for i in range(2,33):
    #     configs.append([i,pure,None,None,budget, -1, False, False, False])


    # for mult in [2,4,6,8,12,16]:
    #     for pop in [2,4,6,8,12,16]:
    #         configs.append([pop*mult,best_k(1.0/mult),None,gp, budget, 200, False, False, False])

    # for model in [rbf([1], 10.0),rbf([5], 5.0),elm(10),elm(30), mlp([100]),mlp([1,1,1]),mlp([10,10,10]), mlp([10,10])]:
    #     for pop in [48]:
    #         configs.append([pop,best_k(1.0/8),None,model,budget, 200, False, False,False])

    for config in configs:
        res = single_config(config,problem_info, df=df) 
        storage.store_data(res, None)
        df = pd.concat([df, res], ignore_index=True)

    evo.last_cached_doe = None
    return df
    



def single_config(config,problem_info, df=None):
    pop_size,evo_mode,dim_red, model,budget, train_num, sort_train,scale_train,cma_sees_appoximations = config
    pop_size = int(pop_size)
    (dim_red_f, dim_red_name) = dim_red if dim_red else (None, '')
    (model_f, model_name) = model if model else (None, '')
    
    full_desc = f'{pop_size}_{evo_mode}'+ ('_' if len(dim_red_name)>0 else '') + f'{dim_red_name}' + ('_' if len(model_name)>0 else '') + f'{model_name}_'+f'{note}'
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
            vals = [pop_size,evo_mode,model_name ,dim_red_name, budget, train_num, note]
            assert(len(pd_cols.determining_cols) == len(vals))
            df_filtered = df
            for n,v in zip(pd_cols.determining_cols,vals):
                df_filtered = df_filtered[df_filtered[n]==v]
    
    results = []
    for problem in suite:
        fun, dim, ins = problem.id_triple
        # check if this exact experiment was already run
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

        
        surrogate = models.Surrogate(model_f, dim_red_f, train_num, sort_train, scale_train)
        observer.observe(problem)
        start_time = timer()

        evals, vals, s, o,rxs = evo.optimize(
            problem,
            surrogate,
            pop_size = pop_size, 
            true_evals=budget,
            printing=True, 
            seed= 42,
            surrogate_usage=evo_mode,
            cma_sees_appoximations = cma_sees_appoximations
        )
        end_time = timer()
        problem.free()
        elapsed = end_time - start_time
        timestamp = datetime.now().strftime("%m_%d___%H_%M_%S")

        df_row = [vals, evals, pop_size,evo_mode,model_name,dim_red_name, ins, fun, dim, elapsed, observer.result_folder, timestamp, budget, train_num, sort_train,scale_train,cma_sees_appoximations, note]
        results.append(df_row)

    # list(zip(*l))
    # for problem in suite:  # this loop will take several minutes or longer
    #     run_problem(problem,observer)
    print(f"....................................................................run complete {config}")
   
    res = pd.DataFrame({k:v for (k,v) in zip (pd_cols.all_cols,unzip(results))}) # converts list of dataframe slices to dataframe
    return res



if __name__ == '__main__':
    # ['pure', 'gp', 'elm', rbf]
    # df = storage.merge_and_load()
    # df_og = storage.merge_and_load()
    # df = run(df)
    # plot(df)
    # datastore_store(load_data(),'w')

    # df_og = storage.merge_and_load()
    # df_og = run(df_og)

    from doe2vec import exp_bbob
    