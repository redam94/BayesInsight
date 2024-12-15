import numpy as np

__all__ = ["linear", "exp", "indexp", "power", "log", "saturation", "hill"]


def linear(x):
    return x


def exp(x, alpha: float):
    return 1 - np.exp(-x / alpha)


def indexp(x, alpha: float, mean: float):
    return 1 - np.exp(-x / mean * alpha)


def power(x, alpha: float):
    return x**alpha


def log(x):
    return np.log(x)


def saturation(x, alpha, beta, maxper, max):
    beta = beta / 1e8
    return beta ** (alpha ** (x / (maxper * max) * 100**2))


def hill(x, K, n, mean=None):
    if mean is None:
        mean = x.mean(axis=-1)

    x_ = x / mean[..., None]
    return x_**n / (K**n + x_**n)
