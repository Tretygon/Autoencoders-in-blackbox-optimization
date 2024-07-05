import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize
# https://stackoverflow.com/questions/62376164/how-to-change-max-iter-in-optimize-function-used-by-sklearn-gaussian-process-reg
class MyGPR(GaussianProcessRegressor):
    def __init__(self, max_iter, **kwargs):
        super().__init__( **kwargs)
        self.max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        i = self.max_iter
        
        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)


# from https://towardsdatascience.com/gaussian-process-models-7ebce1feb83d
 


def RBF_kernel(xn, xm, l = 1):
    """
    Inputs:
        xn: row n of x
        xm: row m of x
        l:  kernel hyperparameter, set to 1 by default
    Outputs:
        K:  kernel matrix element: K[n, m] = k(xn, xm)
    """
    K = np.exp(-np.linalg.norm(xn - xm)**2 / (2 * l**2))
    return K
def make_RBF_kernel(X, l = 1, sigma = 0):
    """
    Inputs:
        X: set of φ rows of inputs
        l: kernel hyperparameter, set to 1 by default
        sigma: Gaussian noise std dev, set to 0 by default
    Outputs:
        K:  Covariance matrix 
    """
    K = np.zeros([len(X), len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            K[i, j] = RBF_kernel(X[i], X[j], l)
    return K + sigma * np.eye(len(K))

def gaussian_process_predict_mean(X, y, X_new):
    """
    Inputs:
        X: set of φ rows of inputs
        y: set of φ observations 
        X_new: new input 
    Outputs:
        y_new: predicted target corresponding to X_new
    """
    rbf_kernel = make_RBF_kernel(np.vstack([X, X_new]))
    K = rbf_kernel[:len(X), :len(X)]
    k = rbf_kernel[:len(X), -1]
    return  np.dot(np.dot(k, np.linalg.inv(K)), y)


def plot(train_x, train_y,x, f):
    # Range of x to obtain the confidence intervals.
    x = np.linspace(0, 10, 1000)# Obtain the corresponding mean and standard deviations.
    y_pred = []
    y_std = []
    gpr = GaussianProcessRegressor().fit(train_x, train_y)
    for i in range(len(x)):
        X_new = np.array([x[i]])
        m,s = gpr.predict(X_new.reshape(-1, 1), return_std = True)
        y_pred.append(m)
        y_std.append(np.sqrt(s))
        
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)
    plt.figure(figsize = (15, 5))
    plt.plot(x, f(x), "r")
    plt.plot(train, f(train), "ro")
    plt.plot(x, y_pred, "b-")
    plt.fill(np.hstack([x, x[::-1]]),
            np.hstack([y_pred - 1.9600 * y_std, 
                    (y_pred + 1.9600 * y_std)[::-1]]),
            alpha = 0.5, fc = "b")
    plt.xlabel("$x$", fontsize = 14)
    plt.ylabel("$f(x)$", fontsize = 14)
    plt.legend(["$y = x^2$", "Observations", "Predictions", "95% Confidence Interval"], fontsize = 14)
    plt.grid(True)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.show()




    

if __name__ == '__main__':
    import main
    main.main()