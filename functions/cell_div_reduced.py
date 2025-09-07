import numpy as np
from numba import njit

from sample_set.CFD import CFD
from simplex_grad.simplex_grad import simplex_grad

@njit
def full_path(X, dt, final_T, N0=1., a2=0.3):
    a1, a3, k0, m0 = X

    dt = dt
    num_steps = int(final_T/dt + 0.5)
    
    out = np.zeros(shape=(num_steps, 1))
    out[0, 0] = N0

    for i in range(1, num_steps):
        N0 = out[i - 1, 0]
        
        N0_delta = (a3 - a1 - a2)*N0 - (k0*N0**2)/(1 + m0 * N0)
        N0 = N0 + dt*N0_delta
        out[i, 0] = N0
    
    return out

@njit
def f(X, dt, final_T):
    return full_path(X, dt, final_T)[-1, 0]

@njit
def f_batch(X, dt, final_T):
    out = np.empty(len(X))
    for i in range(len(X)):
        out[i] = f(X[i], dt, final_T)
    return out

class CellDivReduced():
    def __init__(self, d=4, dt=0.01, final_T=100):
        self.dt = np.float64(dt)
        self.final_T = np.float64(final_T)
        
        self.h_f2 = np.float64(1e-6)
        self.h_f1 = np.float64(1e-8)

    def sample_domain_points(self, N, d=4, prct_dev=0.2):
        # params defined in `Sensitivity analysis methods in the biomedical sciences.`
        a1 = np.float64(0.1)
        a3 = np.float64(0.69)
        k0 = np.float64(0.1)
        m0 = np.float64(0.1)

        x_sug = np.array([a1, a3, k0, m0], dtype=np.float64)
        x_samples = (2*np.random.uniform(size=(N, len(x_sug))) - 1).astype(np.float64)
        x_samples = x_sug * (1 + x_samples * prct_dev)
        return x_samples

    def f_batch(self, X):
        return f_batch(X, self.dt, self.final_T)
    
    def f(self, X):
        return f(X, self.dt, self.final_T)
    
    def f1(self, X):
        S_CFD = CFD.gen_sample_set(len(X), 0, h=self.h_f1)
        grad_out = simplex_grad(self, X, S_CFD)
        return grad_out

    def f2(self, X):
        h = self.h_f2
        dim = len(X)
        H = np.zeros(shape=(dim, dim), dtype=np.float64)
        I = np.eye(dim, dtype=np.float64)
        for i in range(dim):
            for j in range(dim):
                H[i, j] = (self.f(X + h * I[i] + h * I[j]) - self.f(X + h * I[i] - h * I[j]) - self.f(X - h * I[i] + h * I[j]) + self.f(X - h * I[i] - h * I[j]))/(4*h**2)
        return H
    
    def __call__(self, X):
        return self.f(X)
