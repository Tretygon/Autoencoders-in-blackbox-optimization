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
matplotlib.use('TkAgg')
os.environ['KERAS_BACKEND'] = 'tensorflow'
suite1 = cocoex.Suite("bbob", "", "")
def main():
    # shutil.rmtree("ppdata")
    # shutil.rmtree("exdata")
    problem_num = 15
    dim = 10
    budget = int(5000/dim)
    problem_info = f"function_indices:{problem_num} dimensions:{dim} instance_indices:1-15"
    configs = [ 
        # [5,Pure(),None,None,'pure5'],
        # [10,Pure(),None,None,'pure10'],
        [15,Pure(),None,None,'pure15'],
        # [20,Pure(),None,None,'pure20'],
        # [50,Best_k(10,1,1),p(models.vae,[1/2]),p(models.elm,200),'vae+elm200'],
        # [50,Best_k(25,1,1),models.id,p(models.gp,GPK.Matern(nu=5/2),),'gp_50_25'],
        # [50,Best_k(10,1,1),models.id,p(models.gp,GPK.Matern(nu=5/2),),'gp_50_10'],
        # [25,Best_k(5,1,1),models.id,p(models.gp,GPK.Matern(nu=5/2),),'gp_25_10'],
        # [10,Best_k(5,1,1),models.id,p(models.gp,GPK.Matern(nu=5/2),),'gp_10_5'],
        # [50,Best_k(10,1,1),p(models.pca,1/2),p(models.elm,200),'pca+elm200'],
        # [50,Best_k(10,1,1),p(models.vae,[1/2]),p(models.rbf_network,[1/2],5/2,),'vae+rbf'],
        # [50,Best_k(10,1,1),models.id,p(models.elm,200),'elm200'],
        # [50,Best_k(10,1,1),p(models.pca,1/2),p(models.gp,GPK.Matern(nu=5/2)),'pca+gp'],
        # [50,Best_k(10,1,1),p(models.vae,[1/2]),p(models.gp,GPK.Matern(nu=5/2)),'vae+gp'],
        # [50,Best_k(10,1,1),p(models.pca,1/2),p(models.rbf_network,[1/2],5/2,),'pca+rbf'],
        # [50,Best_k(10,1,1),models.id,p(models.rbf_network,[1/2],5/2,),'rbf'],
        # [50,Best_k(5,1,1),models.id,p(models.mlp,[1/4]),'mlp'],
    ]
    # out_folders = Parallel(n_jobs=2)(delayed(single_config)(config) for config in configs)
    func_names = ['Sphere','Ellipsoidal','Rastrigin','B ̈uche-Rastrigin','Linear Slope','Attractive Sector','Step Ellipsoidal','Rosenbrock','Rosenbrock rotated','Ellipsoidal','Discus','Bent Cigar','Sharp Ridge','Different Powers','Rastrigin','Weierstrass','Schaffers F7','Schaffers F7 moderately ill-conditioned','Composite Griewank-Rosenbrock F8F2','Schwefel','Gallagher’s Gaussian 101-me Peaks','Gallagher’s Gaussian 21-hi Peaks','Katsuura','Lunacek bi-Rastrigin']
    results = [single_config(config,budget,problem_info) for config in configs]
    fig, ax = plt.subplots()
    ax.set(xlabel='log evals', ylabel='func value',
    title=f'{func_names[problem_num-1]}\n{problem_info}',xscale = 'linear',yscale='linear')
        
    
    fun_mins = []
    for config,config_res in zip(configs,results):
        config_desc = config[-1]
        evals, vals_ = list(map(np.array,zip(*config_res)))
        
        for i,med in enumerate(vals_):
            # med = np.median(vals,0)
            ax.plot(evals[i], med,label=config_desc)
            fun_mins.append(med[-1])
            # print([a[-1] for a in med])
    order = np.argsort(fun_mins)
    handles, labels = plt.gca().get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(fun_mins[idx],2)) for idx in order])
    ax.grid()
    graphs = os.listdir('graphs')
    ord = int(graphs[-1].split('.')[0]) + 1 if any(graphs) else 0
    fig.savefig(f"graphs/{ord}.png")
    plt.show()
    ### post-process data
    # cocopp.genericsettings.isConv = True
    # cocopp.main(' '.join(out_folders))
    # cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    # webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

def single_config(config,budget,problem_info):
    pops,surrs,dim_red,model, desc = config
    opts = {
        'algorithm_name': desc,
        'algorithm_info': '"ieaieaiea"',
        "result_folder": desc
    }
    suite = cocoex.Suite("bbob", "", problem_info) # or eg instance_indices:1-5
    minimal_print = cocoex.utilities.MiniPrint()
    observer = cocoex.Observer("bbob", opts)
    all_ran_funcs = set()
    # results = Parallel(n_jobs=8, prefer="threads")(delayed(p(run_problem,observer,pops,surrs,dim_red,model, desc,trues,minimal_print))(problem) for problem in suite)
    results = [p(run_problem,observer,pops,surrs,dim_red,model, desc,budget,minimal_print)(problem) for problem in suite]
    # list(zip(*l))
    # for problem in suite:  # this loop will take several minutes or longer
    #     run_problem(problem,observer)
    print(f"....................................................................run complete {config}")
    return results

def run_problem(observer,pops,surrs,dim_red,model, desc,trues,printer,problem):
    global suite1
    problem_num,dim = problem.id_function, problem.dimension
    # problem1 = suite1.get_problem_by_function_dimension_instance(problem_num, dim, 1)
    observer.observe(problem)
    print(f'---------------------------------------------------f: {problem_num}  dim: {dim}')
    
    bests = evo.run_surrogate(
                problem,
                problem,
                pop_size = pops, 
                true_evals=int(trues*dim), 
                surrogate_usage=surrs,
                dim_red_f = dim_red,
                model_f = model,
                printing=True
            )
    res = f'{pops}, {trues}, {surrs}, {desc}, {round(np.min(bests),2)}'
    print(res)
    problem.free()
    return bests
    # printer(problem, final=problem_f=='f24')
if __name__ == '__main__':
    main()