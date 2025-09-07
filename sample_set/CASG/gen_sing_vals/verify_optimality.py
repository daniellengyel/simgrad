import numpy as np
import matplotlib.pyplot as plt

from sample_set.CASG.gen_sing_vals.gen_sing_vals import solve_sing_vals_exact
from sample_set.CASG.gen_sing_vals.utils import lmbda_loss, grad_lmbda_loss_proxy_jax

def ill_cond_operator(kappa, v):
    d = len(v)
    v = v / np.linalg.norm(v)
    return np.eye(d) + np.outer(v, v) * (kappa - 1)

def run_comp(configs, N=10):
    '''
    Run the test on chosen parameter ranges. We add ill-conditioning with ill_cond_operator.
    '''
    l_exact_indef = {}
    l_exact_def = {}
 
    for conf in configs:
        d, kappa, sig, h = conf['d'], conf['kappa'], conf['sig'], conf['h']

        l_exact_indef[(d, kappa, sig, h)] = []
        l_exact_def[(d, kappa, sig, h)] = []

        for _ in range(N):
            # Get Symmetric Quadratic
            Q = np.random.normal(size=(d, d))
            Q = (Q + Q.T)/2.

            # Add ill conditioning
            v = np.random.normal(size=d)
            T = ill_cond_operator(kappa**0.5, v)
            Q_T = T.T @ Q @ T

            # Get diagonal for sing_val solver
            D = np.linalg.eigh(Q_T)[0]
            if np.sum(D) < 0:
                D = -D
            D = np.sort(D)
            
            # Solve for sing_vals and assess optimality for indef
            sing_vals_exact = solve_sing_vals_exact(D, sig, h)
            loss = lmbda_loss(sing_vals_exact**2, D, sig)
            grad_norm = np.linalg.norm(grad_lmbda_loss_proxy_jax(sing_vals_exact**2, D, sig)[sing_vals_exact < h - 1e-6]) / d
            l_exact_indef[(d, kappa, sig, h)].append([loss, grad_norm])

            # Solve for sing_vals and assess optimality for def
            D = np.sort(np.abs(D))
            sing_vals_exact = solve_sing_vals_exact(D, sig, h)
            loss = lmbda_loss(sing_vals_exact**2, D, sig)
            grad_norm = np.linalg.norm(grad_lmbda_loss_proxy_jax(sing_vals_exact**2, D, sig)[sing_vals_exact < h - 1e-6]) / d
            l_exact_def[(d, kappa, sig, h)].append([loss, grad_norm])

    return l_exact_indef, l_exact_def

if __name__ == "__main__":
    main_config = {
        "sigs": [1, 1e-2, 1e-4, 1e-6],
        "hs": [1, 1e-1, 1e-2, 1e-3, 1e-4],
        "ds": [2, 4, 8, 16, 32],
        "kappas": [0.01, 0.1, 1, 10, 100, 1000]
    }
    
    configs = get_confs(main_config) 


    l_exact_indef, l_exact_def = run_comp(configs, N=10)

    grad_vals_indef = np.array(list(l_exact_indef.values()))[:, :, 1].flatten()
    grad_vals_def = np.array(list(l_exact_def.values()))[:, :, 1].flatten()
    grad_vals = np.concatenate([grad_vals_indef, grad_vals_def])

    print("Gradient Norm Stats")
    print("Median", np.median(grad_vals))
    print("Mean", np.mean(grad_vals))
    print("Std", np.std(grad_vals))
    print("25th Percentile", np.percentile(grad_vals, 25))
    print("75th Percentile", np.percentile(grad_vals, 75))
    print("Max", np.max(grad_vals))
    print("Min", np.min(grad_vals))


    # Plot grad vals
    plt.hist(np.log10(grad_vals[grad_vals > 1e-12]), bins = 100)
    plt.show()

