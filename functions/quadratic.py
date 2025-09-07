import numpy as np
from numba import njit

@njit
def f(Q, b, c, x):
    return x.T @ Q @ x + b.T @ x + c

@njit
def f_batch(Q, b, c, X):
    X = np.ascontiguousarray(X)  # Ensure C-contiguous layout
    out = np.empty(len(X))
    for i in range(len(X)):
        out[i] = f(Q, b, c, X[i])
    return out

class Quadratic:
    def __init__(self, Q, b, c, sig=0,  noise_type="gaussian"):
        self.Q = 0.5 * (Q + Q.T)
        self.b = b
        self.c = c
        self.sig = sig
        self.noise_type = noise_type
    
    def f(self, x):
        return f(self.Q, self.b, self.c, x)
    
    def f_batch(self, X):
        """X.shape = (N, d)"""
        return f_batch(self.Q, self.b, self.c, X)
    

    def f_tilde(self, x):
        out = self.f(x)
        if self.noise_type == "uniform":
            eps = self.sig * np.sqrt(3)
            return out + 2 * eps * np.random.uniform() - eps
        else:
            return out + self.sig * np.random.normal() 
    
    def f_batch_tilde(self, X):
        out = self.f_batch(X)
        if self.noise_type == "uniform":
            eps = self.sig * np.sqrt(3)
            return out + 2 * eps * np.random.uniform(shape=(len(X),)) - eps
        else:
            return out + self.sig * np.random.normal(shape=(len(X),)) 
    
    def f1(self, x):
        return np.array(2 * self.Q @ x + self.b)
    
    def f1_batch(self, X):
        return np.array(2 * X @ self.Q + self.b)

    def f2(self, X):
        return np.array(2 * self.Q)
    
    def __call__(self, x):
        return self.f(x)