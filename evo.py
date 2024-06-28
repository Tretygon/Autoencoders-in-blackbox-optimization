import numpy as np
from cmaes import CMA,SepCMA,CMAwM,get_warm_start_mgd
import tensorflow as tf
#cocoex.solvers.random_search
from sklearn.decomposition import PCA
import math
import progress_bar
from tqdm import tqdm

def run_surrogate(problem, pop_size, true_evals, surrogate_evals_per_true, new_model_f, max_model_uses=1,warm_start_task = None,printing=True):
    bounds = np.stack([problem.lower_bounds,problem.upper_bounds],axis=0).T
   
    if warm_start_task != None:
        source_solutions = []
        for _ in range(1000):
            x = np.random.random(warm_start_task.dimension)
            value = warm_start_task(x)
            source_solutions.append((x, value))
        # ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(true_points, gamma=0.5, alpha=0.1)
        # optimizer = CMA(mean=ws_mean, sigma=ws_sigma,cov=ws_cov,bounds=bounds,population_size=pop_size)
    else:
        optimizer = CMA(mean=np.zeros(problem.dimension), sigma=1.3,bounds=bounds,population_size=pop_size,seed=42,n_max_resampling=10)   

    
    current_model_uses = 0
    all_points = []
    true_xs= []
    true_ys= []

    # assert((problem.lower_bounds<0).all())
    # rand_points = np.random.random_sample([1000,problem.dimension])
    # rand_points = rand_points*(np.abs(problem.lower_bounds)+problem.upper_bounds) - np.abs(problem.lower_bounds)
    # ys = [problem(x) for x in rand_points]
    # true_points += list(zip(rand_points,ys))
    # pca = PCA(300).fit(rand_points)
    # print(pca.explained_variance_ratio_)
    surrogate_ever_needed = any(map(lambda a: a > 0, surrogate_evals_per_true))
    surr_eval_i = 0
    best = 9999999999
    true_evals_left = true_evals
    surrogate_uses_left = 0
    generation = 0
    while true_evals_left > 0:
        generation += 1
        use_true_f = not(surrogate_uses_left > 0) 

        solutions = []
        pop = []
        for _ in range(pop_size):
            x = optimizer.ask()
            pop.append(x)
        
        
        if use_true_f:
            ys = [problem(x) for x in pop]
            true_xs += pop
            true_ys += ys
            ys = np.array(ys)
            ys_t = ys
            # surrogate is generated at generation 1 and after every being used max_model_uses times; if only the true function is ever used, the model is never generated 
            if surrogate_ever_needed:
                surrogate=new_model_f(true_xs,true_ys) 
        else :
            y = surrogate(np.array(pop))
            ys = tf.unstack(y,axis=0)
            ys_t = np.array([problem(x) for x in pop])


        best_in_gen =np.min(ys)
        best_gen_true = np.min(ys_t)
        best = min(best,best_in_gen)         

        avg_err = 10 #np.average(np.abs(ys_t - ys))

        if printing:
            print(f"{generation} ,,,, {round(best_gen_true, 2)}", f' ,,,, {round(best_in_gen, 2)} ,,,,, {round(avg_err, 2)}' if not use_true_f else '')
        optimizer.tell(list(zip(pop,ys)))
        # print(pca.explained_variance_ratio_.sum())
        # a = pca.transform(pop)
        # print(a.shape)
        # optimizer.tell([(x,-y) for (x,y) in solutions])


        if use_true_f:
            true_evals_left -= 1
            progress_bar.progress_bar(true_evals-true_evals_left,true_evals)
            surrogate_uses_left += surrogate_evals_per_true[surr_eval_i]
            surr_eval_i = (surr_eval_i + 1) % len(surrogate_evals_per_true)
        else:
            current_model_uses += 1
            surrogate_uses_left -= 1

            
    
    res = min(ys_t)
    print(' '*80,end='\r')
    if printing:
        print(f'Final:  {res}')
    return res










if __name__ == '__main__':
    import main
    main.main()