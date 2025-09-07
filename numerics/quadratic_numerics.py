import numpy as np

from functions.quadratic import Quadratic

from numerics.config_utils import generate_configs
from numerics.utils import construct_df, get_loss, plot_figs_exact_loss

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pickle
import os

from tqdm import tqdm

def ill_cond_operator(kappa, v):
    d = len(v)
    v = v / np.linalg.norm(v)
    return np.eye(d) + np.outer(v, v) * (kappa - 1)

def run_comp(conf, N=1000):
    np.random.seed(os.getpid()) # since every worker gets same seed at first
    
    d, kappa, sig, h = conf['d'], conf['kappa'], conf['sig'], conf['h']
    loss_def, loss_indef = [], []

    for _ in range(N):
        # Get Symmetric Quadratic
        D = np.eye(d)
        D_indef = np.ones(d)
        D_indef[:d // 2] = -1
        D_indef = np.diag(D_indef)

        # rotate matrix
        Q = np.linalg.qr(np.random.randn(d, d))[0]
        Q_def = Q.T @ D @ Q
        Q_indef = Q.T @ D_indef @ Q

        # Add ill conditioning
        v = np.random.normal(size=d)
        T = ill_cond_operator(kappa**0.5, v)
        Q_def = T.T @ Q_def @ T
        Q_indef = T.T @ Q_indef @ T

        x_0 = np.zeros(shape=d) 
        # Solve for sing_vals and assess optimality for indef
        F_indef = Quadratic(Q_indef, np.zeros(shape=d), 0, sig=0)
        curr_loss_indef = get_loss(F_indef, sig, h, x_0)

        # Solve for sing_vals and assess optimality for def
        F_def = Quadratic(Q_def, np.zeros(shape=d), 0, sig=0)
        curr_loss_def = get_loss(F_def, sig, h, x_0)

        loss_indef.append(curr_loss_indef)
        loss_def.append(curr_loss_def)

    return loss_indef, loss_def    



if __name__ == "__main__":
    '''
    Run the test on chosen parameter ranges. We add ill-conditioning with ill_cond_operator.
    '''
    load_res = True
    if load_res:
        with open('numerics/results/quad_res.pkl', 'rb') as f:
            main_config, N, l_indef, l_def = pickle.load(f)
        configs = generate_configs(main_config)
    else:
        N=1000
        main_config = {
            "sig": [1e-3],
            "h": [1],
            "d": [2, 16, 18, 19],
            "kappa": list(np.logspace(-5, 5, 11))
        }

        configs = generate_configs(main_config)
        print("Generated Configs. Num", len(configs)) 

        fixed_N_run = partial(run_comp, N=N)
        with ProcessPoolExecutor(max_workers=8) as executor:
            pool_res = list(tqdm(executor.map(fixed_N_run, configs), total=len(configs)))

        l_indef = [p[0] for p in pool_res]
        l_def = [p[1] for p in pool_res]

        with open('numerics/results/quad_res.pkl', 'wb') as f:
            pickle.dump([main_config, N, l_indef, l_def], f)

    l_indef_df = construct_df(configs, l_indef)
    l_def_df = construct_df(configs, l_def)

    plot_figs_exact_loss(l_indef_df, main_config, "quad_indef", "numerics")
    plot_figs_exact_loss(l_def_df, main_config, "quad_def", "numerics")

