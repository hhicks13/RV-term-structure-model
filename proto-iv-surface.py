import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm
from math import factorial
from scipy.optimize import minimize
import statsmodels.api as sm

#import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#init
np.random.seed(0)
WINDOW = 24*60

#globals 
RV_d = np.array(1)
RV_w = np.array(1)
RV_m = np.array(1)

#
# Majewski et al. 
#
# t here is a day
#
# as long as function is defined with numpy, gradient is attainable
#
def neg_log_like(params):
    global RV_d
    global RV_w
    global RV_m
    
    delta = params[0]
    theta = params[1]
    d = params[2]
    beta_d = params[3]
    beta_w = params[4]
    beta_m = params[5]
    alpha_d = params[6]
    alpha_w = params[7]
    alpha_m = params[8]
    gamma = params[9]
    
    T = len(RV_d)
    epsilon = np.random.normal(0,1,size=len(RV_d))
    
    # leverage terms
    def l_d(t):
        return (epsilon[t] - gamma*np.sqrt(RV_d[t]))**2
    
    # leverage term weekly
    def l_w(t):
        l_w = 0
        for i in range(1,7):
            l_w += l_d(t-i)
        return (1/6)*l_w
    
    # leverage term monthly 
    def l_m(t):
        l_m = 0
        for i in range(24,30):
            l_m+=l_d(t-i)
        return (1/24)*l_m

    # Linear Terms
    def O(t):
        return d + beta_d*RV_d[t] + beta_w*RV_w[t] + beta_m*RV_m[t] + alpha_d*l_d(t) + alpha_w*l_w(t) + alpha_m*l_m(t)
    
    # Left term of likelihood function
    f0 = 0
    for i in range(31,T):
        f0 = RV_d[i]/theta + O(i-1)

    # Right Inner term of likelihood function
    def f2(t):
        f1 = 0
        for k in range(1,90):
            A = RV_d[t]**(delta+k-1)
            B = (theta**(delta+k))*(factorial(delta+k-1))
            C = O(t-1)**(k)
            D = factorial(k)
            f1 += (A/B)*(C/D)
        return f1

    # Right Outer term of likelihood function
    f3 = 0
    for j in range(31,T):
        f3 += np.log(f2(j))

    LL = -f0 + f3
    return -LL


#
#
#
def main():

    #
    # fix timestamp indexing
    #
    
    data = pd.read_csv('bitstamp/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
    data = data[10000:4800000]
    data = data.reset_index(drop=True)
    print(data.info())
    data['Date'] = [datetime.fromtimestamp(x) for x in data['Timestamp']]
    data = data.drop([ "Volume_(Currency)", "Timestamp"], axis=1)
    print(data.isnull().any())
    data = data.dropna()
    data['log_r'] = np.log(data.Weighted_Price) - np.log(data.Weighted_Price.shift(1))

    #
    # new columns and clean
    #
    
    data['RV_d'] = np.sqrt(np.square(data['log_r']).rolling(WINDOW).sum())
    data['RV_w'] = sum([data['RV_d'].shift(i*WINDOW) for i in range(1,7)])*(1/6)
    data['RV_m'] = sum([data['RV_d'].shift(i*WINDOW) for i in range(7,30)])*(1/24)
    print(data[:800600])
    print("drop na")
    print(data.info())
    data = data.dropna()

    #
    # set global vars
    #
    global RV_d
    RV_d = data.RV_d
    global RV_w
    RV_w = data.RV_w
    global RV_m
    RV_m = data.RV_m
    
    # subsample 24 hour periods
    RV_d = np.array(RV_d[::WINDOW])
    RV_w = np.array(RV_w[::WINDOW])
    RV_m = np.array(RV_m[::WINDOW])
    # sanity check
    assert len(RV_d) == len(RV_w) == len(RV_m), "lengths are incorrect"

    #
    # MLE gradient
    #

    # number of parameters
    syms = ['delta','theta','d','beta_d','beta_w','beta_m','alpha_d','alpha_w','alpha_m','gamma']
    K = 10
    _jacobian = jacobian(neg_log_like)
    _hessian = hessian(neg_log_like)
    param_start = np.append(np.ones(K-1),1.0)
    result1 = minimize(neg_log_like, param_start, method = 'BFGS',options={'disp': False}, jac = _jacobian)
    print("Convergence? ", result1.success)
    print("Num Evals: ", result1.nfev)
    print("")
    for i,sym in enumerate(syms):
        print(f"{sym :<10} ~ ~ ~ ~ {param_start[i]:>20.3f}")
    print("")



if __name__ == "__main__":
    main()
