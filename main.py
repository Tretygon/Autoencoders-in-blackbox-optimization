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
import pandas as pd
# import tensorflow as tf
from sklearn.decomposition import PCA
# import tensorflow_addons as tfa
import sklearn.gaussian_process.kernels as GPK
import progress_bar
import math
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
matplotlib.use('TkAgg')
os.environ['KERAS_BACKEND'] = 'tensorflow'
suite1 = cocoex.Suite("bbob", "", "")
scales = []

unzip = lambda a:list(map(list,list(zip(*a))))
pd_cols = ['vals', 'evals', 'pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim', 'full_desc'] # and 'ranks', '
def main():
    # shutil.rmtree("ppdata")
    # shutil.rmtree("exdata")
    problem_num = 2
    dim = 10
    budget = int(3*10e1*dim)#int(20000/dim)
    problem_info = f"function_indices:{problem_num} dimensions:{dim} instance_indices:1"

    pure = Pure()
    best_k = lambda a: Best_k(a,1,1)
    
    id = models.id, 'id'
    vae = lambda n: (p(models.vae,[n]), f'vae{n}')
    pca = lambda n: (p(models.pca,n), f'pca{n}')
    
    elm = lambda n: (p(models.elm,n), f'elm{n}')
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
        # [16*2,best_k(1.0/2),vae(1/2),elm(200)],
        # [16*2,best_k(1.0/2),vae(1/2),gp],
        [16*22,best_k(1.0/2),pca(1/2),gp],
        # [16*2,best_k(1.0/2),id,rbf([1/2], 5/2)],
    ]
    # out_folders = Parallel(n_jobs=2)(delayed(single_config)(config) for config in configs)
    func_names = ['Sphere','Ellipsoidal','Rastrigin','B ̈uche-Rastrigin','Linear Slope','Attractive Sector','Step Ellipsoidal','Rosenbrock','Rosenbrock rotated','Ellipsoidal','Discus','Bent Cigar','Sharp Ridge','Different Powers','Rastrigin','Weierstrass','Schaffers F7','Schaffers F7 moderately ill-conditioned','Composite Griewank-Rosenbrock F8F2','Schwefel','Gallagher’s Gaussian 101-me Peaks','Gallagher’s Gaussian 21-hi Peaks','Katsuura','Lunacek bi-Rastrigin']
    df_slices,out_folders = unzip([single_config(config,budget,problem_info) for config in configs])
    df = pd.concat(df_slices)
    print(df)
    df, common_eval = get_ranking(df)

    
    ### manual ploting 
    # fig, ax = plt.subplots()
    # ax.set(xlabel='evals', ylabel='func value',
    # title=f'{func_names[problem_num-1]}\n{problem_info}',xscale = 'linear',yscale='linear')    
    # fun_mins = []
    # for config,config_res in zip(configs,results):
    #     config_desc = config[-1]
    #     evals, vals = list(map(np.array,zip(*config_res)))
    #     box_plt_x = list(range(1,len(evals[0])+1))
        
    #     med = np.median(vals,0)
    #     # ax.plot(list(range(1,len(evals[0])+1)), med,label='median',linestyle='dotted',zorder=10)
    #     ax.plot(box_plt_x, med,label='median',linestyle='dotted',zorder=10,color='black')
    #     fun_mins.append(med[-1])
    #     for i,val in enumerate(vals):
    #         ax.plot(box_plt_x, val,label=config_desc)
    #         fun_mins.append(val[-1])
    #     ax.boxplot(vals,vert=True)  
    #     ax.set_xticks(box_plt_x,labels=list(evals[0]))
    # order = np.argsort(fun_mins)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(fun_mins[idx],2)) for idx in order])
    # ax.grid()
    # graphs = os.listdir('graphs')
    # ord = int(graphs[-1].split('.')[0]) + 1 if any(graphs) else 0
    # fig.savefig(f"graphs/{ord}.png")
    # plt.show()

    fig, ax = plt.subplots()
    ax.set(xlabel='log evals', ylabel='rank',
    title=f'{func_names[problem_num-1]}\n{problem_info}',xscale = 'linear',yscale='linear')
        
    eaea df.groupby(['full_desc'])['ranks'].apply   #needs to be a custom lambda to take a series of arrays and average them on axis 1
    for (desc, ranks) in :
        ax.plot(common_eval, ranks,label=desc, linestyle='', marker='|')

    rank_avg = np.average(np.array(df['ranks'].tolist()),axis=1)
    order = np.argsort(rank_avg)

    handles, labels = plt.gca().get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(rank_avg[idx],3)) for idx in order])
    ax.grid()
    graphs = os.listdir('graphs')
    ord = int(graphs[-1].split('.')[0]) + 1 if any(graphs) else 0
    fig.savefig(f"graphs/{ord}.png")
    plt.show()


# fig, ax = plt.subplots()
#     ax.set(xlabel='log evals', ylabel='func value',
#     title=f'{func_names[problem_num-1]}\n{problem_info}',xscale = 'linear',yscale='linear')
        
    
#     fun_mins = []
#     for config,config_res in zip(configs,results):
    
#         config_desc = config[-1]
#         evals, vals_ = list(map(np.array,zip(*config_res)))
#         # ii = np.argmax(evals[0]>100)
#         # evals = evals[:,ii:]
#         # vals_ = vals_[:,ii:]
#         for i,med in enumerate(vals_):
#             # med = np.median(vals,0)
#             ax.plot(evals[i], med,label=config_desc)
#             fun_mins.append(med[-1])
#             # print([a[-1] for a in med])
#     order = np.argsort(fun_mins)
#     handles, labels = plt.gca().get_legend_handles_labels()
#     ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(fun_mins[idx],2)) for idx in order])
#     ax.grid()
#     graphs = os.listdir('graphs')
#     ord = int(graphs[-1].split('.')[0]) + 1 if any(graphs) else 0
#     fig.savefig(f"graphs/{ord}.png")
#     # plt.show()

    ### coco plotting 
    cocopp.genericsettings.isConv = True
    cocopp.main(' '.join(out_folders))
    # cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

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

    # results = Parallel(n_jobs=8, prefer="threads")(delayed(p(run_problem,observer,pops,surrs,dim_red,model, desc,trues,minimal_print))(problem) for problem in suite)
    results = []
    for problem in suite:
        fun, dim, ins = problem.id_triple
        res = p(run_problem,observer,pop_size,evo_mode,dim_red_f,model_f, full_desc,budget,minimal_print)(problem) 
        vals,evals = res
        df_row = [evals, vals, pop_size,evo_mode,model_name,dim_red_name, ins, fun, dim, full_desc]
        results.append(df_row)
    # list(zip(*l))
    # for problem in suite:  # this loop will take several minutes or longer
    #     run_problem(problem,observer)
    print(f"....................................................................run complete {config}")
    unzipped = unzip(results)
    return pd.DataFrame({k:v for (k,v) in zip (pd_cols,unzipped)}),observer.result_folder

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

# replaces absolute measurents by relative order => model performance can now be compared across different functions and instances
def get_ranking(df:pd.DataFrame):
    only_evals = df['evals']
    all_steps = [e[2] - e[1] for e in only_evals] 
    master_step = min(all_steps)
    max_eval = min([e[-1] for e in only_evals])
    
    master_evals = np.array(range(max(all_steps),max_eval+1,master_step))
    

    def vals_to_correct_sampling(run):
        (evals,vals) = run['evals'], run['vals']
        cur_step = evals[2] - evals[1]
        begin_i = np.nonzero(evals>=master_evals[0])[0][0] #all evals should start the same
        evals,vals = evals[begin_i:],vals[begin_i:]
        k = int(cur_step/master_step)
        exact = master_step*k == cur_step #bez zaokrouhlovani
        if exact and k == 1:
            res = vals
        elif exact and k <1:
            res = vals[::k] #take each k-th item 
        elif exact and k > 1:
            res = [a for a in vals for _ in range(k)] #duplicate each item k-times
        else:
            res = []
            cur_i = 0
            for master_eval in master_evals:
                while len(evals) < cur_i and master_eval > evals[cur_i]:
                    cur_i+=1
                if master_eval > evals[cur_i]: break #end of arr
                res.append(vals[cur_i])
        return np.array(res)

    transformed = df.apply(vals_to_correct_sampling,axis=1) 
    end_i = min([len(a) for a in transformed])
    transformed = [a[:end_i] for a in transformed] #all evals should end the same
    master_evals = master_evals[:end_i]
    ranks = np.apply_along_axis(lambda a: a.argsort().argsort()+1,0,np.array(transformed)) # turn columns to ranks
    df['normalised_len_vals'] = transformed
    df['ranks'] = list(ranks)

    return df, master_evals
if __name__ == '__main__':
    main()