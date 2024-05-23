import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


# from https://towardsdatascience.com/gaussian-process-models-7ebce1feb83d

def get_regressor(train, f):
    y = f(train)
    gpr = GaussianProcessRegressor().fit(train, y)
    return gpr

def step(train, x, f):
    #X = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    gpr = get_regressor(train,f)
    mean, std = gpr.predict(x.reshape(-1, 1), return_std = True) ## can also compute covariance
    return mean, std


def plot(train,f,x):
    # Range of x to obtain the confidence intervals.
    x = np.linspace(0, 10, 1000)# Obtain the corresponding mean and standard deviations.
    y_pred = []
    y_std = []
    gpr = get_regressor(train,f)
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