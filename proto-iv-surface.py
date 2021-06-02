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
import itertools as it

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

    #T = len(RV_d)
    T = 41
    epsilon = np.random.normal(0,1,size=len(RV_d))
    truncation = 15

    # leverage terms
    def l_d(t,gamma):
        return (epsilon[t] - gamma*np.sqrt(RV_d[t]))**2
    
    # leverage term weekly
    def l_w(t,gamma):
        l_w = 0
        for i in range(1,7):
            l_w += l_d(t-i,gamma)
        return (1/6)*l_w
    
    # leverage term monthly 
    def l_m(t,gamma):
        l_m = 0
        for i in range(24,30):
            l_m+=l_d(t-i,gamma)
        return (1/24)*l_m

    # heterogenous parameter
    def O(t,beta_d,beta_w,beta_m,alpha_d,alpha_w,alpha_m,gamma):
        # parabolic leverage assumption
        d = 0
        return d + beta_d*RV_d[t] + beta_w*RV_w[t] + beta_m*RV_m[t] + alpha_d*l_d(t,gamma) + alpha_w*l_w(t,gamma) + alpha_m*l_m(t,gamma)

    #
    # factorial which can be used by autograd
    #
    def _factorial(n):
        val=1
        while n>=1:
            val = val * n
            n = n-1
        return  val

    #
    # inner term on right side of likelihood equation
    #
    def f2(t,delta,theta,beta_d,beta_w,beta_m,alpha_d,alpha_w,alpha_m,gamma):
        f1 = 0
        d = 0
        for k in range(1,truncation):
            A = np.power(RV_d[t],(delta+k-1))
            B = np.power(theta,delta+k)*_factorial(delta+k-1)
            C = np.power(d + beta_d*RV_d[t] + beta_w*RV_w[t] + beta_m*RV_m[t] + alpha_d*l_d(t,gamma) + alpha_w*l_w(t,gamma) + alpha_m*l_m(t,gamma),k)
            D = factorial(k)
            f1 += (A/B)*(C/D)
        return f1

    #
    # MLE
    #
    def neg_log_like(params):    
        theta = params[0]
        delta = params[1]
        beta_d  = params[2]
        beta_w = params[3]
        beta_m = params[4]
        alpha_d = params[5]
        alpha_w = params[6]
        alpha_m = params[7]
        gamma = params[8]
        # parabolic leverage assumption (P-LHARG)
        d = 0
        # leverage terms
        # Linear Terms
        # Left term of likelihood function
        f0 = 0
        T = 40
        for i in range(T,T+1000):
            f0 += RV_d[i]/theta + d + beta_d*RV_d[i-1] + beta_w*RV_w[i-1] + beta_m*RV_m[i-1] + alpha_d*l_d(i-1,gamma) + alpha_w*l_w(i-1,gamma) + alpha_m*l_m(i-1,gamma)
        
        # Right Outer term of likelihood function
        f3 = 0
        for j in range(T,T+1000):
            f3 += np.log(f2(j-1,delta,theta,beta_d,beta_w,beta_m,alpha_d,alpha_w,alpha_m,gamma))

        LL = -f0 + f3
        return -LL
    
    # number of parameters
    syms = ['theta','delta','beta_d','beta_w','beta_m','alpha_d','alpha_w','alpha_m','gamma']
    K = 10
    print("Truncation: ",truncation)
    print("T: ",T)
    print("<><><> jacobian calculation")
    _jacobian = jacobian(neg_log_like)
    print("<><><> jacobian computed")
    _hessian = hessian(neg_log_like)

    #
    # try grid of points 
    #
    scales = [1e00,1e01,1e02,1e03]
    test_points = [point for point in it.product(scales,repeat=9)]
    min_val = neg_log_like(test_points[0])
    min_candidate = test_points[0]
    print("first test",min_candidate)
    for i,point in enumerate(test_points):
        print(f"testing {i}/{len(test_points)}       min_val: {min_val}, min_candidate: {min_candidate}")
        cur_val = neg_log_like(point)
        if cur_val < min_val:
            min_candidate = point
            min_val = cur_val

    print(min_candidate)
    print(min_val)


    print("number of test points",len(test_points))

    
    #param_start = np.array([1.068e-002,
    #                        1.243e000,
    #                        2.429e004,
    #                        2.317e004,
    #                        1.322e04,
    #                        2.376e-001,
    #                        1.194e-001,
    #                        3.85e-001,
    #                        2.237e002])
    #result1 = minimize(neg_log_like, param_start, method = 'Newton-CG',options={'disp': True}, jac = _jacobian)
    #print("Convergence? ", result1.success)
    #print("Num Evals: ", result1.nfev)
    #print("")
    #for i,x in enumerate(result1.x):
    #    print(f"{syms[i] :<10} ~ ~ ~ ~ {x:>20.3f}")
    #print("")



if __name__ == "__main__":
    main()
