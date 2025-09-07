from scipy.interpolate import RBFInterpolator

import numpy as np

import jax.numpy as jnp
from jax import jacfwd


class RBF:
    def __init__(self, X, F_vals, smoothing):
        self.smoothing = smoothing
        self.rbf = RBFInterpolator(X, F_vals, smoothing=smoothing, kernel="cubic")
        self.coeffs = jnp.array(self.rbf._coeffs)
        self.y = jnp.array(self.rbf.y)
        self.shift = jnp.array(self.rbf._shift)
        self.scale = jnp.array(self.rbf._scale)
        self.powers = jnp.array(self.rbf.powers)


    def f(self, X):
        return self._evaluate(X)

    def f1(self, X):
        p = self.y.shape[0]

        r = jnp.linalg.norm(X - self.y, axis=1)
        out = 3 * self.coeffs[:p, 0] @ (r.reshape(-1, 1) * (X - self.y)) + self.coeffs[p + 1:, 0]/self.scale
        return out
        

    def f2(self, X): 
        y = self.y        
        p = self.y.shape[0]

        r = jnp.linalg.norm(X - y, axis=1)

        a = 3 * jnp.eye(len(X)) * (self.coeffs[:p, 0] @ r)
        R = (X - self.y).T * self.coeffs[:p, 0]/r
        R = R * (r!=0)
        b = 3 * R @ (X - self.y)

        out = a + b
        return out

    def f3(self, X, verbose=False): 
        d = len(X)
        D3 = jacfwd(self.f2)(X)
        if not np.any(np.isnan(D3)):
            if verbose:
                print("RBF D3: Autodiff")
            return np.array(D3)

        if verbose:
            print("RBF D3: Finite Diff")
            
        # backup
        D3 = np.zeros((d, d, d))
        h = 1e-3
        I = np.eye(d)
        for i in range(d):
            D3[i, :, :] = (self.f2(X + I[i] * h) - self.f2(X - I[i] * h))/(2 * h)

        return D3
    
    def _evaluate(self, X):
        y = self.y
        p = y.shape[0]

        xhat = (X - self.shift)/self.scale

        r = jnp.linalg.norm(X - y, axis=1)
        kernel_vec = r**3
        
        poly_vec = jnp.prod(xhat ** self.powers, axis=1)

        out = kernel_vec @ self.coeffs[:p, 0] + poly_vec @ self.coeffs[p:, 0]

        return out





