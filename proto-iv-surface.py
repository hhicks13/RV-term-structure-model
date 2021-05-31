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

np.random.seed(0)
#
# Majewski et al. 
#
# t here is a day

def neg_log_like(delta,theta,d,beta_d,beta_w,beta_m,alpha_d,alpha_w,alpha_m,_gamma,RV_d_t,RV_w_t,RV_m_t):
    # subsample 24 hour periods
    window = 24*60
    RV_d_t = np.array(RV_d_t[::window])
    RV_w_t = np.array(RV_w_t[::window])
    RV_m_t = np.array(RV_m_t[::window])
    GAMMA = lambda x : factorial(x-1)
    T = len(RV_d_t)
    epsilon = np.random.normal(0,1,size=len(RV_d_t))
    
    # leverage terms
    l_d = lambda t : (epsilon[t] - _gamma*np.sqrt(RV_d_t[t]))^2
    l_w = lambda t : (1/6)*sum([l_d(t - i) for i in range(1,7)])
    l_m = lambda t : (1/24)*sum([l_d(t- i) for i in range(1,30)])

    # eq 3.3
    O = lambda t: d + beta_d*RV_d_t[t] + beta_w*RV_w_t[t] + beta_m*RV_m_t[t] + alpha_d*l_d_t(t) + alpha_w*l_w_t(t) + alpha_m*l_m_t(t)

    # must make sure t is not too small (for l_m)
    terms = lambda t : sum([ (RV_d_t[t]**(delta+k-1)/(theta**(delta+k)*GAMMA(delta+k)))*((O(t-1)**(k))/(np.factorial(k))) for k in range(1,90)])

    LL = -sum([(RV_d_t[i]/theta) + O(i-1) for i in range(1,T)]) + sum([ np.log(terms(j)) for j in range(1,T)])
    return -LL


#
#
#
def main():
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
    #
    #
    

    window = 60*24 # day long window

    
    data['RV_d'] = np.sqrt(np.square(data['log_r']).rolling(window).sum())
    data['RV_w'] = sum([data['RV_d'].shift(i*window) for i in range(1,7)])*(1/6)
    data['RV_m'] = sum([data['RV_d'].shift(i*window) for i in range(7,30)])*(1/24)
    print(data[:800600])
    print("drop na")
    print(data.info())
    data = data.dropna()
    data['e_t'] = np.random.normal(0,1,size=3217332)
    #data['l_d'] = np.random.normal(0,1) - 
    #data['l_w']
    #data['l_m']
    RV_d_t = data.RV_d
    RV_w_t = data.RV_w
    RV_m_t = data.RV_m
    
    print(data[:800600])

    


if __name__ == "__main__":
    main()
