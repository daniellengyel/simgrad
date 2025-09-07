import numpy as np
from numba import njit

import jax.numpy as jnp
from jax import grad, hessian

# Putting outside of class since numba cannot work with self as input
# Felt cleaner just putting outside of class instead of ignoring input

@njit
def f(X):
    d = len(X)
    a = np.exp(-0.2 * np.sqrt(1. / d * np.square(np.linalg.norm(X))))
    b = - np.exp(1. / d * np.sum(np.cos(2 * np.pi * X)))
    out = -20 * a + b + 20 + np.exp(1)
    return out

@njit
def f_batch(X):
    out = np.empty(len(X))
    for i in range(len(X)):
        out[i] = f(X[i])
    return out

def f_jax(X):
    d = X.shape[0]
    a = jnp.exp(-0.2 * jnp.sqrt(1. / d * jnp.square(jnp.linalg.norm(X))))
    b = -jnp.exp(1. / d * jnp.sum(jnp.cos(2 * jnp.pi * X)))
    return -20 * a + b + 20 + jnp.exp(1)

class Ackley:
    def __init__(self, d=None):
        self._f1 = grad(lambda x: f_jax(x))
        self._f2 = hessian(lambda x: f_jax(x))

    def f(self, x):
        return f(x)
    
    def f_batch(self, X):
        """X.shape = (N, d)"""
        return f_batch(X)

    def f1(self, X):
        return np.array(self._f1(X))
    
    def f2(self, X):
        return np.array(self._f2(X).reshape(X.size, X.size))
    
    def sample_domain_points(self, N, d):
        """
        Sample N points from the Ackley function domain.
        
        Args:
            N: Number of points to sample
            d: Dimension of the domain (if None, uses self.d)
        
        Returns:
            X: Array of shape (N, d) with sampled points
        """        
        # Sample from uniform distribution on [-0.5, 0.5]^d
        # This covers the typical domain where Ackley function is interesting
        X = np.random.uniform(size=(N, d)) - 0.5
        return X
    
    def __call__(self, x):
        return self.f(x)
