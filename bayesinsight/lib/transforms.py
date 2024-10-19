import pandas as pd
import numpy as np

def linear(x):
    return x

def exp(x, alpha: float):
    return 1 - np.exp(-x/alpha)

def indexp(x, alpha: float, mean: float):
    return 1 - np.exp(-x/mean*alpha)

def power(x, alpha: float):
    return x**alpha

def log(x):
    return np.log(x)

def hill(x, K, n, mean=None):
    if mean is None:
        mean = x.mean(axis=-1)
        
    x_ = x/mean[...,None]
    return x_**n/(K**n + x_**n)