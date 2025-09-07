import numpy as np
import pandas as pd
from tqdm import tqdm 
import os


from numerics.config_utils import generate_configs
from numerics.utils import construct_df, get_loss_rbf
from functions.cell_div_reduced import CellDivReduced
from functions.quadratic import Quadratic
from global_model.RBF import RBF

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from numerics.utils import plot_figs_N_RBF, pre_compute_pts


def run_comp(configs, points_cache, rbf_cache):
    np.random.seed(os.getpid()) # since every worker gets same seed at first

    l_res = []
    l_quad_res = []

    F = CellDivReduced()

    for conf in configs:
        
        d, sig, h, N_RBF = conf['d'], conf['sig'], conf['h'], conf['N_RBF']

        x_key = (d, sig)
        xs = points_cache[x_key]

        rbf_key = (d, sig, N_RBF)
        F_RBF = rbf_cache[rbf_key]

        curr_l_list = []
        curr_l_quad_list = []
        for x_0 in xs:
            # Loss for point
            curr_loss = get_loss_rbf(F, F_RBF, sig, h, x_0)
            curr_l_list.append(curr_loss)

            # Loss for quadratic
            F_Quad = Quadratic(1/2. * F.f2(x_0), np.zeros(shape=d), 0, sig=0)
            curr_loss_quad = get_loss_rbf(F_Quad, F_RBF, sig, h, x_0)
            curr_l_quad_list.append(curr_loss_quad)

        l_res.append(curr_l_list)
        l_quad_res.append(curr_l_quad_list)

    return construct_df(configs, l_res), construct_df(configs, l_quad_res)


if __name__ == "__main__":
    import pickle
    load_res = False
    if load_res:
        with open('numerics/results/cell_div_res.pkl', 'rb') as f:
            main_config, N, df, df_quad = pickle.load(f)
        
    else:
        main_config = {
            "sig": [1e-3, 1e-5],
            "h": [1, 1e-1, 1e-2, 1e-3],
            "d": [4], 
            "N_RBF": list(int(x) for x in np.logspace(1, np.log(501)/np.log(10), 11))
        }

        N = 500

        configs = generate_configs(main_config)
        print("Generated Configs. Num", len(configs)) 

        # ===== partition by (d, sig, N_RBF) =====
        conf_by_key = {}
        for conf in configs:
            d, sig, N_RBF = conf['d'], conf['sig'], conf['N_RBF']
            curr_key = (d, sig, N_RBF)
            if curr_key not in conf_by_key:
                conf_by_key[curr_key] = []
            conf_by_key[curr_key].append(conf)

        # ===== pre-compute caches before parallel execution =====
        points_cache, rbf_cache = pre_compute_pts(CellDivReduced, configs, N)

        # ===== run parallel on (d, sig, N_RBF) =====
        fixed_N_run = partial(run_comp, points_cache=points_cache, rbf_cache=rbf_cache)
        configs = list(conf_by_key.values())

        with ProcessPoolExecutor(max_workers=12) as executor:
            pool_res = list(tqdm(executor.map(fixed_N_run, configs), total=len(configs)))

        df = pd.concat([p[0] for p in pool_res])
        df_quad = pd.concat([p[1] for p in pool_res])

        df = df.reset_index()
        df_quad = df_quad.reset_index() 

        with open('numerics/results/cell_div_res.pkl', 'wb') as f: 
            pickle.dump([main_config, N, df, df_quad], f)

    plot_figs_N_RBF("CellDiv", df, "numerics")
