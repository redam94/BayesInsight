import pandas as pd
import numpy as np

def linear(x):
    return x

def exp(x, alpha: float):
    return 1 - np.exp(-x/alpha)

def indexp(x, alpha: float, index_to: float):
    return 1 - np.exp(-x/index_to*alpha)

def power(x, alpha: float):
    return x**alpha

def log(x):
    return np.log(x)

def hill(x, K, n):
    mean = x.mean(axis=-1)
    x_ = x/mean[...,None]
    return x_**n/(K**n + x_**n)