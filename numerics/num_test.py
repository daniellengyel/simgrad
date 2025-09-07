import numpy as np

from functions.quadratic import Quadratic

from numerics.config_utils import generate_configs
from numerics.utils import construct_df, get_loss, plot_figs_exact_loss

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pickle
import os

from tqdm import tqdm

def get_FFD(H, h, sig):
    H_diag = np.abs(np.diag(H))
    S = np.diag(np.where(H_diag != 0, 2 * np.sqrt(sig / np.abs(H_diag)), np.inf))
    S = np.minimum(S, h)

    






def run_comp(conf, N=1000):
    np.random.seed(os.getpid()) # since every worker gets same seed at first
    
    d, kappa, sig, h = conf['d'], conf['kappa'], conf['sig'], conf['h']

    
    # Get Symmetric Quadratic
    D = np.eye(d)
    D[-1,-1] = kappa

    x_0 = np.zeros(shape=d) 

    # Solve for sing_vals and assess optimality for def
    F_def = Quadratic(D, np.zeros(shape=d), 0, sig=0)
    loss_def = get_loss(F_def, sig, h, x_0)

    return loss_def    



if __name__ == "__main__":
    '''
    Run the test on chosen parameter ranges. We add ill-conditioning with ill_cond_operator.
    '''
    N=10
    main_config = {
        "sig": [1e-3],
        "h": [1],
        "d": [2, 16, 18, 19],
        "kappa": list(np.logspace(-5, 5, 20))
    }
    
    configs = generate_configs(main_config)
    print("Generated Configs. Num", len(configs)) 

    fixed_N_run = partial(run_comp, N=N)
    pool_res = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fixed_N_run, conf) for conf in configs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            pool_res.append(future.result())


    # for conf in configs:
    #     pool_res.append(fixed_N_run(conf))

    
    pool_res = np.array(pool_res)
    l_indef = pool_res[:, 0, :]
    l_def = pool_res[:, 1, :]

    l_indef_df = construct_df(configs, l_indef)
    l_def_df = construct_df(configs, l_def)

    with open('numerics/results/quad_res.pkl', 'wb') as f:
        pickle.dump([main_config, N, l_indef, l_def], f)

    numerics_path = "/Users/daniellengyel/Projects/simgrad/numerics"

    plot_figs_exact_loss(l_indef_df, main_config, "quad_indef", numerics_path)
    plot_figs_exact_loss(l_def_df, main_config, "quad_def", numerics_path)


    