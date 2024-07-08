import numpy as np
from cmaes import CMA,SepCMA,CMAwM,get_warm_start_mgd
import tensorflow as tf
#cocoex.solvers.random_search
from sklearn.decomposition import PCA
import math
import progress_bar
from tqdm import tqdm
from typing import Union,List
from dataclasses import dataclass
from scipy import stats





@dataclass
class Alternate_full_generations:
    sur_gens_per_true: List[int]

@dataclass
class Best_k:
    k: Union[int,float]
    retrain_every: int
    generate_multiplier: float

@dataclass
class Pure: 
    pass

def get_surrogate(train_x, train_y,model_f,dim_red_f,old_model,old_dim_red):
    X = np.array(train_x)
    Y = np.array(train_y) 
    
    # k = 1
    # n = len(Y)
    # weights = 1/(k*n + (np.arange(n)))
    # s = np.argsort(Y)
    # # chosen_i = np.random.choice(np.argsort(Y),size=len(Y),p=weights/np.sum(weights))
    # chosen_i = s[:int(len(Y))]
    # np.random.shuffle(chosen_i)
    # X = X[chosen_i,:]
    # Y = Y[chosen_i]
    weights = None
  
    dim_red = dim_red_f(X,Y,weights,old_dim_red)
    latentX = dim_red(X)
    model = model_f(latentX,Y,weights,old_model)
    def run(xs):
        ys = model(dim_red(xs))
        return ys#,model,dim_red

    def ansa(ms,comb, xs):
        ys = [m(xs) for m in ms]
        ys = np.stack(ys,0)
        ys = np.min(ys,0)



    return run,dim_red, model



def run_surrogate(problem, pop_size, true_evals, surrogate_usage:Union[Alternate_full_generations, Best_k, Pure],dim_red_f, model_f,printing=True):
    bounds = np.stack([problem.lower_bounds,problem.upper_bounds],axis=0).T
   
    # if warm_start_task != None:
    #     source_solutions = []
    #     for _ in range(1000):
    #         x = np.random.random(warm_start_task.dimension)
    #         value = warm_start_task(x)
    #         source_solutions.append((x, value))
    #     # ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(true_points, gamma=0.5, alpha=0.1)
    #     # optimizer = CMA(mean=ws_mean, sigma=ws_sigma,cov=ws_cov,bounds=bounds,population_size=pop_size)
    # else:
    optimizer = CMA(mean=np.zeros(problem.dimension), sigma=1.3,bounds=bounds,population_size=pop_size,seed=42,n_max_resampling=10)   

    
    current_model_uses = 0
    all_points = []
    true_xs= []
    true_ys= []
    model,dim_red = None, None
    # mean_weights = []
    if isinstance(surrogate_usage,Best_k):

        eval_best_k = surrogate_usage.k if isinstance(surrogate_usage.k,int) else int(pop_size * surrogate_usage.k)
        retrain_rate = surrogate_usage.retrain_every
       
        optimizer = CMA(mean=np.zeros(problem.dimension), sigma=1.3,bounds=bounds,population_size=eval_best_k,seed=42,n_max_resampling=10)   

        best = 9999999999
        true_evals_left = true_evals - eval_best_k
        generation = 0
        for _ in range(3):
            for _ in range(int(eval_best_k)):
                x = optimizer.ask()
                y = problem(x)
                true_xs.append(x)
                true_ys.append(y)
            optimizer.tell(list(zip(true_xs[-eval_best_k:],true_ys[-eval_best_k:])))
        while true_evals_left > 0:
            if generation % retrain_rate == 0:
                surrogate,dim_red,model = get_surrogate(true_xs,true_ys,model_f,dim_red_f,model,dim_red)
            xs = []
            for _ in range(int(pop_size*abs(surrogate_usage.generate_multiplier))):
                x = optimizer.ask()
                xs.append(x)

            ys = surrogate(np.array(xs)) 
            sorted_i = tf.argsort(ys).numpy()
            xs,ys = np.array(xs)[sorted_i], np.array(ys)[sorted_i]
            
            if False and len(pop) > pop_size: ## doesn't work
                z = stats.zscore(ys)
                sorted_i = np.argsort(z)
                pop,ys = pop[sorted_i],ys[sorted_i]
                z_coef = 1.5
                mask = np.abs(z)<z_coef
                while np.sum(mask) < pop_size:
                    z_coef *= 1.1
                    mask = np.abs(z)>z_coef
                    if z_coef > 10:
                        print()
                aaa = [problem(x) for x in pop[np.logical_not(mask)]]
                pop,ys = pop[mask], ys[mask]
            xs,ys = xs[:pop_size], ys[:pop_size]
            k = min(true_evals_left,eval_best_k)
            top_k_xs = xs[:k]
            top_k_ys = np.array([problem(x) for x in top_k_xs])
            true_evals_left -= k
            true_xs += list(top_k_xs)
            true_ys += list(top_k_ys)
            # mean_weights *= 0.95
            # mean_weights += [1] * len(top_k_ys)
            


            best_in_gen =np.min(ys)
            best_gen_true = np.min(top_k_ys)
            best = min(best,best_gen_true)         
            avg_err = np.average(np.abs(top_k_ys - ys[:k]))

            if printing:
                print(f"{generation} ,,,, {round(best_gen_true, 2)}", f' ,,,, {round(best_in_gen, 2)} ,,,,, {round(avg_err, 2)}')
            res_y = np.concatenate([top_k_ys,ys[k:]]) 
            res_x = xs 
            # optimizer.tell(list(zip(res_x,res_y))) # full or only true?? or overgenerate and 
            # optimizer.tell(list(zip(xs,ys))) # full or only true?? or overgenerate and 
            optimizer.tell(list(zip(top_k_xs,top_k_ys))) # full or only true?? or overgenerate and 
        
            progress_bar.progress_bar(true_evals-true_evals_left,true_evals)
            generation += 1
        print(' '*80,end='\r') #deletes progress bar
        if printing:
            print(f'Final:  {best}')
        return best,np.min(top_k_ys)

    elif isinstance(surrogate_usage, Pure):
        best = 9999999999
        true_evals_left = true_evals
        generation = 0
        while true_evals_left > 0:
            pop = []
            for _ in range(pop_size):
                x = optimizer.ask()
                pop.append(x)
            ys = [problem(x) for x in pop]
            true_evals_left -= pop_size
            best_in_gen =np.min(ys)
            best = min(best,best_in_gen)        
            if printing:
                print(f"{generation}", f' ,,,, {round(best_in_gen, 2)} ,,,,, {round(best, 2)}')
            optimizer.tell(list(zip(pop,ys))) # full or only true?? or overgenerate and 
        
            progress_bar.progress_bar(true_evals-true_evals_left,true_evals)
            generation += 1
        print(' '*80,end='\r') #deletes progress bar
        if printing:
            print(f'Final:  {best}')
        return best, np.min(ys)

    elif isinstance(surrogate_usage, Alternate_full_generations):
        surrogate_evals_per_true = surrogate_usage.sur_gens_per_true
        surrogate_ever_needed = any(map(lambda a: a > 0, surrogate_evals_per_true))
        surr_eval_i = 0
        best = 9999999999
        true_evals_left = true_evals
        surrogate_gens = 0
        generation = 0
        while true_evals_left > 0:
            generation += 1
            do_true_gen = not(surrogate_gens > 0) 
            pop = []
            for _ in range(pop_size):
                x = optimizer.ask()
                pop.append(x)
            
            
            if do_true_gen:
                ys = [problem(x) for x in pop]
                true_xs += pop
                true_ys += ys
                ys = np.array(ys)
                ys_t = ys
                # surrogate is generated at generation 1 and after every being used max_model_uses times; if only the true function is ever used, the model is never generated 
                
                
                true_evals_left -= pop_size
                surrogate_gens += surrogate_evals_per_true[surr_eval_i]
                surr_eval_i = (surr_eval_i + 1) % len(surrogate_evals_per_true)
                if surrogate_gens > 0:
                    surrogate=get_surrogate(true_xs,true_ys,model_f, dim_red_f) 
            else :
                y = surrogate(np.array(pop))
                ys = tf.unstack(y,axis=0)
                ys_t = np.array([problem(x) for x in pop])
                current_model_uses += 1
                surrogate_gens -= 1


            best_in_gen =np.min(ys)
            best_gen_true = np.min(ys_t)
            best = min(best,best_gen_true)         

            avg_err = np.average(np.abs(ys_t - ys))

            if printing:
                print(' '*80,end='\r') #deletes progress bar
                print(f"{generation} ,,,, {round(best_gen_true, 2)}", f' ,,,, {round(best_in_gen, 2)} ,,,,, {round(avg_err, 2)}' if not do_true_gen else '')
            optimizer.tell(list(zip(pop,ys)))
            if do_true_gen:
                progress_bar.progress_bar(true_evals-true_evals_left,true_evals)
                

                
        
        
        print(' '*80,end='\r') #deletes progress bar
        if printing:
            print(f'Final:  {best}')
        return (best, np.min(true_ys))

    
    else: raise Exception('unknown surrogate_usage')



    










if __name__ == '__main__':
    import main
    main.main()