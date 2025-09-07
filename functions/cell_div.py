import numpy as np
from numba import njit

from sample_set.CFD import CFD
from simplex_grad.simplex_grad import simplex_grad

@njit
def full_path(X, dt, final_T, N0_init, N1_init, N2_init):
    a1, a2, a3, b1, b2, b3, gamma, k0, m0, k1, m1 = X

    dt = dt
    num_steps = int(final_T/dt + 0.5)
    
    out = np.zeros(shape=(num_steps, 3))
    out[0, :] = [N0_init, N1_init, N2_init]

    for i in range(1, num_steps):
        N0, N1, N2 = out[i - 1]
        
        N0_delta = (a3 - a1 - a2)*N0 - (k0*N0**2)/(1 + m0 * N0)
        N1_delta = (b3 - b1 - b2)*N1 + a2*N0 - (k1 * N1**2)/(1 + m1*N1) + (k0 * N0**2)/(1 + m0 * N0)
        N2_delta = -gamma*N2 + b2*N1 + (k1 * N1**2)/(1 + m1*N1)

        N0 = N0 + dt*N0_delta
        N1 = N1 + dt*N1_delta
        N2 = N2 + dt*N2_delta
        out[i, :] = [N0, N1, N2]
    return out

@njit
def f(X, dt, final_T, N0_init, N1_init, N2_init, out_idx):
    return full_path(X, dt, final_T, N0_init, N1_init, N2_init)[-1, out_idx]

@njit
def f_batch(X, dt, final_T, N0_init, N1_init, N2_init, out_idx):
    out = np.empty(len(X))
    for i in range(len(X)):
        out[i] = f(X[i], dt, final_T, N0_init, N1_init, N2_init, out_idx)
    return out

class CellDiv():
    def __init__(self, N0_init=1, N1_init=100, N2_init=100, 
                 dt=0.01, final_T=100, 
                 output_var="N1"):
        self.N0_init = np.float64(N0_init)
        self.N1_init = np.float64(N1_init)
        self.N2_init = np.float64(N2_init)
        self.dt = np.float64(dt)
        self.final_T = np.float64(final_T)
        
        self.output_var = output_var

        self.h_f2 = np.float64(1e-6)
        self.h_f1 = np.float64(1e-8)

    def sample_domain_points(self, N, d=11):
        # params defined in `Sensitivity analysis methods in the biomedical sciences.`
        a1 = np.float64(0.1)
        a2 = np.float64(0.3)
        a3 = np.float64(0.69)
        b1 = np.float64(0.1)
        b2 = np.float64(0.3)
        b3 = np.float64(0.397)
        gamma = np.float64(0.139)
        k0 = np.float64(0.1)
        m0 = np.float64(0.1)
        k1 = np.float64(0.0003)
        m1 = np.float64(0.0004)

        x_sug = np.array([a1, a2, a3, b1, b2, b3,
                          gamma, k0, m0, k1, m1], dtype=np.float64)
        x_samples = (2*np.random.uniform(size=(N, len(x_sug))) - 1).astype(np.float64)
        x_samples = x_sug * (1 + x_samples * 0.1)
        return x_samples

    def f_batch(self, X):
        return f_batch(X, self.dt, self.final_T, 
                       self.N0_init, self.N1_init, self.N2_init, 
                       int(self.output_var[-1]))
    
    def f(self, X):
        return f(X, self.dt, self.final_T, 
                       self.N0_init, self.N1_init, self.N2_init, 
                       int(self.output_var[-1]))
    
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
    
    
    