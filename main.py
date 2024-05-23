import numpy as np
from cmaes import CMA
import cocopp
import cocoex
#cocoex.solvers.random_search

def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)
    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)


def run_surrogate(f,generations,f_true, new_model_f, new_model_every_n_uses,true_eval_every_n):
    optimizer = CMA(mean=np.zeros(f.dim), sigma=1.3)
    current_model_uses = 0
    for generation in range(generations):
        if current_model_uses % new_model_every_n_uses == 0:
            surrogate=new_model_f()
        use_true_f = generation % true_eval_every_n == 0
        current_model_uses += (int)(not use_true_f)
            
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            if use_true_f:
                y = f_true(x)
            else :
                y = surrogate(x)
            solutions.append((x, y))
            print(f"#{generation}, {x}, {value}")
        optimizer.tell(solutions)
