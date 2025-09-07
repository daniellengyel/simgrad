import numpy as np

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

def lmbda_loss(lmbdas, D_diag, sig):    
    if any(lmbdas == 0):
        return np.inf
    d = len(D_diag)
    a = D_diag @ lmbdas
    l_max = np.max(lmbdas)
    return a**2 / (d * l_max) + sig**2 * d / l_max + sig**2 * np.sum(1/lmbdas)

def lmbda_loss_proxy(lmbdas, D_diag, sig):    
    '''The minimizer of this function equals the minimizer of lmbda_loss as showen in [TODO].
    The problem is stricly convex.'''
    if any(lmbdas <= 0):
        return np.inf
    d = len(D_diag)
    a = D_diag @ lmbdas
    l_0 = lmbdas[0]
    return a**2 / (d * l_0) + sig**2 * d / l_0 + sig**2 * np.sum(1/lmbdas)

def lmbda_loss_proxy_jax(lmbdas, D_diag, sig):    
    d = len(D_diag)
    a = jnp.dot(D_diag, lmbdas)
    l_0 = lmbdas[0]

    # We rely on JAX's auto-diff to handle edge cases carefully
    return a**2 / (d * l_0) + sig**2 * d / l_0 + sig**2 * jnp.sum(1.0 / lmbdas)

def grad_lmbda_loss_proxy_jax(lmbdas, D_diag, sig):    
    lmbdas = jnp.array(lmbdas)
    lmbda_grad = jax.grad(lambda lam: lmbda_loss_proxy(lam, D_diag, sig))
    return lmbda_grad(lmbdas)