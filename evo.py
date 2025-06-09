import numpy as np
from cmaes import CMA,SepCMA,CMAwM
import cma
# from cma.purecma import CMAES
# from cma_custom import CMA
import tensorflow as tf
#cocoex.solvers.random_search
from sklearn.decomposition import PCA
import math
import progress_bar
from tqdm import tqdm
from typing import Union,List
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler

import cocoex
from cma.fitness_models import SurrogatePopulation
from doe2vec.doe2vec import doe_model 
from doe2vec.bbobbenchmarks import instantiate
@dataclass
class Alternate_full_generations:
    sur_gens_per_true: List[int]
    def __str__(self):
        return f'Alternate{repr(self.sur_gens_per_true)}'

@dataclass
class Best_k:
    k: Union[int,float]
    def __str__(self):
        return f'BestK{self.k}' 

@dataclass
class Pure: 
    def __str__(self):
        return 'Pure'    

@dataclass
class Doe: 
    follow_up: Union[Best_k, Pure]
    dim: int
    power: int
    latent: int
    def __str__(self):
        return self.__class__.__name__+'_' + str(self.follow_up) +'&'+ f'{self.dim}_{self.power}_{self.latent}'



class Problem: 
    f: callable
    def __str__(self):
        return self.__class__.__name__+str(self.model)
            
# type AAA = Union[Alternate_full_generations, Best_k, Pure, DOE]

last_cached_doe = None
def optimize(problem, surrogate, pop_size, true_evals, surrogate_usage:Union[Alternate_full_generations, Best_k, Pure, Doe], cma_sees_appoximations=False,printing=True,seed = 42, optimizer= None):
    # bounds = np.stack([problem.lower_bounds,problem.upper_bounds],axis=0).T
    optimizer_popsize = pop_size
    def new_optim(dim=problem.dimension, optimizer_popsize=optimizer_popsize):
        # initial = np.random.rand(problem.dimension)*9 - 4.5
        initial = np.zeros(dim)
        return CMA(mean=initial, sigma=1.0, seed=seed,bounds=np.array([[-5.0,5.0]]*dim),population_size=optimizer_popsize)
    next_specimens_forced = []
    def next_gen(size=optimizer_popsize):
        nonlocal next_specimens_forced
        forced = next_specimens_forced[:size]
        next_specimens_forced = next_specimens_forced[size:]
        xs = np.array(forced+[optimizer.ask() for _ in range(size-len(forced))])
        return xs
        
    # if warm_start_task != None:
    #     source_solutions = []
    #     for _ in range(1000):
    #         x = np.random.random(warm_start_task.dimension)
    #         value = warm_start_task(x)
    #         source_solutions.append((x, value))
    #     # ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(true_points, gamma=0.5, alpha=0.1)
    #     # optimizer = CMA(mean=ws_mean, sigma=ws_sigma,cov=ws_cov,bounds=bounds,population_size=pop_size)
    # else:
       # should there be popsize or K???
    current_model_uses = 0
    evals_wihout_change = 0
    true_xs= []
    true_ys= []
    true_evals_left = true_evals 
    best = 9999999999
    best_x = np.zeros(problem.dimension)
    overall_best = 9999999999
    overall_best_x = 9999999999
    bests,bests_evals = [],[] #best found values overall and timestamps of currently used evaluations
    insert_best = False
    def eval_true(xs):
        nonlocal true_evals_left,true_evals,bests,bests_evals,printing,best,problem,true_xs,true_ys,evals_wihout_change,optimizer,overall_best,best_x,overall_best_x
        ys = np.array([problem(x) for x in xs])
        ys = np.where(np.isinf(ys), 1e11, ys)
        ys = np.where(np.isnan(ys), 1e11, ys)

        true_xs += list(xs)
        true_ys += list(ys)
        true_evals_left -= xs.shape[0]
        if true_evals_left < 0:
            print()
        if best > np.min(ys):
            index = np.argmin(ys)
            best = ys[index]
            best_x = xs[index]
            if overall_best > best:
                overall_best = best
                overall_best_x = best_x
            evals_wihout_change = 0
        else:
            evals_wihout_change += xs.shape[0]
            if evals_wihout_change > 2000:
                optimizer = new_optim()
                best = 9999999999
        bests.append(overall_best)
        bests_evals.append(true_evals-true_evals_left)
        if printing:
            progress_bar.progress_bar(overall_best,true_evals-true_evals_left,true_evals)
        if printing and true_evals_left <= 0:
            print(' '*80,end='\r') #deletes progress bar
        return ys

    generation = 0
    # mean_weights = []
    if False:
        es = cma.CMAEvolutionStrategy ( problem.dimension * [0.1], 0.1 )
        surrogate = cma.fitness_models.SurrogatePopulation(problem)
        while not es.stop():
            X = es.ask() # sample a new population
            F = surrogate( X ) # see Algorithm 1
            es.tell(X , F ) # update sample distribution
            es.inject([ surrogate.model.xopt ])
            es.disp() # just checking what 's going one
        return es.best.f
    if isinstance(surrogate_usage, Doe):
        global last_cached_doe
        if optimizer == None:
            optimizer = new_optim(optimizer_popsize=optimizer_popsize)
        if last_cached_doe == None or not (last_cached_doe.dim==surrogate_usage.dim and last_cached_doe.m == surrogate_usage.power and last_cached_doe.latent_dim==surrogate_usage.latent):
            last_cached_doe = doe_model(dim=surrogate_usage.dim,m=surrogate_usage.power, latent_dim=surrogate_usage.latent)
            last_cached_doe.load_or_train()
        doe = last_cached_doe
        fun, dim, ins = problem.id_triple
        problem_info = f"function_indices:{fun} dimensions:{dim} instance_indices:{ins}"
        suite = cocoex.Suite("bbob", "", problem_info) 
        for p in suite: #only one item
            # y = eval_true(np.clip(doe.sample*10 - 5,-4.99, 4.99))
            y = np.array([p(x) for x in np.clip(doe.sample*10 - 5,-4.99, 4.99)])
            mx = np.nanmax(y[y != np.inf])
            y = np.where(np.isinf(y), mx, y)
            y = np.where(np.isnan(y), mx, y)

        func_approx_, dist = doe.func_approx(y)
        func_approx = lambda x: func_approx_(x,False)
        func_approx.dimension = problem.dimension
        e, v, s, o, rxs= optimize(func_approx, surrogate, pop_size, int(3e2), Pure())
        # next_specimens_forced = [o.ask() for _ in range(2*pop_size)]
        next_specimens_forced = rxs[-2*pop_size:]
        surrogate_usage = surrogate_usage.follow_up
        #continue with another surrogate_usage branch

    if isinstance(surrogate_usage,Best_k):
        generated_population = pop_size * surrogate_usage.k
        eval_best_k = pop_size 
        optimizer_popsize = eval_best_k
        if optimizer == None:
            optimizer = new_optim(optimizer_popsize=optimizer_popsize)
        # retrain_rate = surrogate_usage.retrain_every
        # surrogate = SurrogatePopulation(problem)
        # optimizer.inject()
        
        
        # unreported = 0
        # unreported_offset = 0
        # for _ in range(0):
        #     xs = np.array([optimizer.ask() for _ in range(int(eval_best_k))])
        #     ys = eval_true(np.array(true_xs)[:-eval_best_k])
        #     unreported += eval_best_k
        #     if optimizer.population_size <= unreported:
        #         optimizer.tell(list(zip(true_xs[unreported_offset:unreported_offset+optimizer.population_size],true_ys[unreported_offset:unreported_offset+optimizer.population_size])))
        #         unreported -= optimizer.population_size
        for _ in range(1):
            xs = next_gen(optimizer_popsize)
            ys = eval_true(xs)
            # if optimizer.population_size == surrogate_usage.k:
            optimizer.tell(list(zip(xs,ys))) 
        surrogate.train(true_xs,true_ys)
        while true_evals_left > 0:
            xs = next_gen(generated_population)
            ys = surrogate(xs) 
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
            xs,ys = xs[:generated_population], ys[:generated_population] 
            k = min(true_evals_left,eval_best_k)
            top_k_xs = xs[:k]  
            top_k_ys = eval_true(top_k_xs)
            surrogate.train(true_xs,true_ys)

            if true_evals_left <= 0:
                pass # the training is at the end, no need to tell the optimizer anything anymore 
                     # also throws errors bc the final population size may be inconpatible with the one the optimizer has 
            # elif False and generation == 0 and optimizer.population_size == pop_size:
            #     unreported_x = np.array(true_xs)[:-(k+unreported)]
            #     unreported_y = np.array(true_ys)[:-(k+unreported)]
            #     res_y = np.concatenate([unreported_y,ys[len(unreported_y):]])
            #     res_x = np.concatenate([unreported_x,xs[len(unreported_x):]])
            #     optimizer.tell(list(zip(res_x,res_y))) 
            elif cma_sees_appoximations and generated_population > surrogate_usage.k: # report the entire population, not just the true evaluated
                ys = surrogate(xs) #extra surrogate eval to guarantee the freshest approximations
                res_y = np.concatenate([top_k_ys,ys[k:]])
                optimizer.tell(list(zip(xs,res_y))) 
            else:
                if insert_best:
                    m = np.argmax(top_k_ys) # replace the current worst with so far best
                    top_k_ys[m] = best
                    top_k_xs[m] = best_x
                optimizer.tell(list(zip(top_k_xs,top_k_ys))) 

            

            avg_err = np.average(np.abs(top_k_ys - ys[:k]))

                # print(f"{generation} ,,,, {round(best_gen_true, 2)}", f' ,,,, {round(best_in_gen, 2)} ,,,,, {round(avg_err, 2)}')
            
            generation += 1    
        
        

    if isinstance(surrogate_usage, Pure):
        if optimizer == None:
            optimizer = new_optim(optimizer_popsize=pop_size)
        while true_evals_left > 0 :
            xs = next_gen()
            if xs.shape[0] > true_evals_left:
                xs = xs[:true_evals_left]
            ys = eval_true(xs)
            if printing:pass
                # print(f"{generation}", f' ,,,,, {round(best, 2)}')
            
            if true_evals_left > 0:
                optimizer.tell(list(zip(xs,ys))) 
            generation += 1

    if isinstance(surrogate_usage, Alternate_full_generations):
        if optimizer == None:
            optimizer = new_optim(optimizer_popsize=optimizer_popsize)
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

    
    if len(bests)==0: raise Exception('unknown surrogate_usage')

    return np.array(bests_evals), np.array(bests), surrogate, optimizer, true_xs



    










if __name__ == '__main__':
    import main
    main.main()