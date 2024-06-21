import numpy as np
from cmaes import CMA,SepCMA,CMAwM,get_warm_start_mgd
import tensorflow as tf
#cocoex.solvers.random_search



def run_surrogate(problem, pop_size, generations, new_model_f, max_model_uses,true_eval_every_n,warm_start_task = None):
    bounds = np.stack([problem.lower_bounds,problem.upper_bounds],axis=0).T
   
    if warm_start_task != None:
        source_solutions = []
        for _ in range(1000):
            x = np.random.random(warm_start_task.dimension)
            value = warm_start_task(x)
            source_solutions.append((x, value))
        ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(true_points, gamma=0.5, alpha=0.1)
        optimizer = CMA(mean=ws_mean, sigma=ws_sigma,cov=ws_cov,bounds=bounds,population_size=pop_size)
    else:
        optimizer = CMA(mean=np.zeros(problem.dimension), sigma=1.3,bounds=bounds,population_size=pop_size)   

    
    current_model_uses = 0
    all_points = []
    true_points = []
    best = 9999999999
    for generation in range(generations):
        
        use_true_f = generation % true_eval_every_n == 0 

        # surrogate is generated at generation 1 and after every being used max_model_uses times; if only the true function is ever used, the model is never generated
        if current_model_uses % max_model_uses == 0 and generation > 0 and true_eval_every_n > 1 and not use_true_f:
            surrogate=new_model_f(true_points) 
            current_model_uses = 0

        
        current_model_uses += (int)(not use_true_f)
            
        solutions = []
        best_in_gen = 1e20
        pop = []
        for _ in range(pop_size):
            x = optimizer.ask()
            pop.append(x)
        
        if use_true_f:
            ys = [problem(x) for x in pop]
        else :
            y = surrogate(np.array(pop))
            ys = tf.unstack(y,axis=0)

        solutions = list(zip(pop,ys))
        if use_true_f:
            true_points += solutions
        all_points += solutions    

        best_gen_true =min(best_in_gen,min(ys))          
        if not use_true_f:
            ys = [problem(x) for x in pop]
        best_in_gen = min(best_in_gen,min(ys))  
        best = min(best,best_in_gen)         

        print(f"{generation} ,,,, {best_gen_true} ,,,, {best_in_gen}")
        optimizer.tell(solutions)
        # optimizer.tell([(x,-y) for (x,y) in solutions])

    












if __name__ == '__main__':
    import main
    main.main()