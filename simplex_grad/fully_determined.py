import numpy as np


def S_loss_quadratic(S, Q, sig):
    dim = len(S)    

    S_inv = np.linalg.inv(S)

    first_term = S_inv.T @ np.diag(S.T @ Q @ S)
    second_term = np.linalg.norm(S_inv, ord="fro")**2
    third_term = S_inv.T @ np.ones(dim)
    third_term = np.linalg.norm(third_term)**2

    res = np.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term)
    return float(res)