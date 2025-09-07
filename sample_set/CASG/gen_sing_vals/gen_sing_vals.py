import numpy as np
from scipy.optimize import minimize

from sample_set.CASG.gen_sing_vals.utils import lmbda_loss_proxy

def solve_sing_vals_cvxpy(D, sig, h, epsilon=1e-6):
    import cvxpy as cp

    if len(D.shape) == 2:
        D = np.diag(D)
    if np.sum(D) < 0:
        D = -D
    
    d = len(D)
    idx_sort = np.argsort(D)
    idx_reverse_sort = np.argsort(idx_sort)
    D_sorted = D[idx_sort]

    # Variables
    lmbda = cp.Variable(d, pos=True)

    # Constraints
    constraints = [
        lmbda >= epsilon,
        lmbda <= h**2,
    ]

    # Objective
    term1 = cp.quad_over_lin(D_sorted @ lmbda, d * lmbda[0])
    term2 = sig**2 * cp.sum(cp.inv_pos(lmbda))
    term3 = sig**2 * d * cp.inv_pos(lmbda[0])
    objective = cp.Minimize(term1 + term2 + term3)

    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, abstol=1e-5, reltol=1e-5, max_iters=1000)

    res = minimize(
        lambda l: lmbda_loss_proxy(l, D, sig),
        x0=lmbda.value,
        method='L-BFGS-B',
        bounds=[(1e-8, h**2)] * d
    )

    return np.sqrt(res.x[idx_reverse_sort])


# ===== Solving Lambda Exactly =====
def get_lambda_j(J, D, sigma, h, c1, c2):
    d = len(D)

    if J == 0:
        C = c1 - np.sqrt(D[0])
        inner = (
                    2 * D[0] * (d + 1)
                    + C**2
                    + C * np.sqrt(8 * D[0] * (d + 1) + C**2)
                )
        a = np.sqrt((d * sigma**2 / (2 * D[0]))) * np.sqrt(inner)
        lambda_val = (d / (2 * a * D[0])) * (a**2 / d + sigma**2 * (d + 1))
    else:
        c1 = sigma * np.sqrt(d * h**2 / 2) * c1
        c2 = h**2 * c2
        disc = c1**2 / 4 - c2**3 / 27

        if disc < 0:
            r = np.sqrt(c2**3 / 27)
            theta = np.acos(c1 / (2 * r))
            a_half = 2 * r**(1/3.) * np.cos(theta / 3)
        else:
            a_half = np.cbrt(c1 / 2 + np.sqrt(disc)) + np.cbrt(c1 / 2 - np.sqrt(disc))
        
        a = a_half**2
        lambda_val = sigma * np.sqrt((d * h**2) / (2 * a * D[J]))

    return lambda_val, a

def solve_sing_vals_exact(D, sigma, h):
    # Setup D to conform to assumptions
    D = D.astype(np.float64)
    d = len(D)

    if np.sum(D) == 0:
        return np.full(d, h**2)
    
    if len(D.shape) == 2:
        D = np.diag(D)
    if np.sum(D) < 0:
        D = -D
    
    d = len(D)
    idx_sort = np.argsort(D)
    idx_reverse_sort = np.argsort(idx_sort)
    D_sorted = D[idx_sort]

    # Set up temp variables
    J = np.sum(D_sorted <= 0) # Due to zero/one index, effectively J + 1 in the pseudocode

    c1 = np.sum(np.sqrt(D_sorted[J:]))
    c2 = np.sum(D_sorted[:J])

    # Loop through the working sets (setting constraints active)
    while J < d:
        lambda_j, a = get_lambda_j(J, D_sorted, sigma, h, c1, c2)
    
        if lambda_j <= h**2:
            break

        c1 -= np.sqrt(D_sorted[J])
        c2 += D_sorted[J]  
        J += 1

    # Populate Lambda
    lambdas = np.empty(d)
    lambdas[:J] = h**2
    if J < d:
        lambdas[J] = lambda_j
        for i in range(J + 1, d):
            lambdas[i] = sigma * np.sqrt((d * lambdas[0]) / (2 * a * D_sorted[i]))

    return np.sqrt(lambdas)[idx_reverse_sort]
