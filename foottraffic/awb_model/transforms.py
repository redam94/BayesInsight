import pandas as pd
import numpy as np

def linear(x):
    return x

def s_shaped(x, alpha, beta, index_to):
    return (beta/(10**10)) ** (alpha ** (x/index_to*100))

def s_origin(x, alpha, beta, index_to):
    return (beta/(10**9)) ** (alpha ** (x/index_to*100)) - (beta/(10**9))

def exp(x, alpha):
    return 1 - np.exp(-x/alpha)

def indexp(x, alpha, index_to):
    return 1 - np.exp(-x/index_to*alpha)

def power(x, alpha):
    return x**alpha

def log(x):
    return np.log(x)

