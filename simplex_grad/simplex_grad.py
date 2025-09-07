import numpy as np

def simplex_grad(f, x_0, S):
    f_S = f.f_batch(S.T + x_0)
    f_x_0 = np.array(f(x_0,))
    return np.linalg.solve(S @ S.T, S @ (f_S - f_x_0))

def simplex_grad_mse(S, grad_est, grad_true, sig):
    N = S.shape[1]
    S_T_inv = np.linalg.inv(S @ S.T) @ S # Pseudo inverse
    first_term = np.linalg.norm(grad_est - grad_true)
    second_term = sig * np.linalg.norm(S_T_inv, ord="fro")
    third_term = sig * np.linalg.norm(S_T_inv @ np.ones(N))
    res = first_term**2 + second_term**2 + third_term**2
    return float(res)