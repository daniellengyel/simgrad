import numpy as np
import pandas as pd
import os

from simplex_grad.simplex_grad import simplex_grad, simplex_grad_mse

import sample_set.CASG.eCASG as eCASG
import sample_set.FFD.FFD as FFD
import sample_set.CFD.CFD as CFD

from global_model.RBF import RBF


# ===== loss for methods =====
def get_loss_rbf(F, F_RBF, sig, h, x_0):
    H = F.f2(x_0)
    Q = 1/2. * H
    
    H_rbf = F_RBF.f2(x_0)
    Q_rbf = 1/2. * H_rbf

    grad_true = F.f1(x_0)

    # FFD 
    S_FFD = FFD.gen_sample_set(Q, sig, h)
    grad_FFD = simplex_grad(F, x_0, S_FFD)
    loss_FFD = simplex_grad_mse(S_FFD, grad_est=grad_FFD, grad_true=grad_true, sig=sig) 
    
    # FFD RBF
    S_FFD_rbf = FFD.gen_sample_set(Q_rbf, sig, h)
    grad_FFD_rbf = simplex_grad(F, x_0, S_FFD_rbf)
    loss_FFD_rbf = simplex_grad_mse(S_FFD_rbf, grad_est=grad_FFD_rbf, grad_true=grad_true, sig=sig) 

    # CASG
    S_CASG = eCASG.gen_sample_set(Q, sig, h)
    grad_CASG = simplex_grad(F, x_0, S_CASG)
    loss_CASG = simplex_grad_mse(S_CASG, grad_est=grad_CASG, grad_true=grad_true, sig=sig)

    # CASG rbf
    S_CASG_rbf = eCASG.gen_sample_set(Q_rbf, sig, h)
    grad_CASG_rbf = simplex_grad(F, x_0, S_CASG_rbf)
    loss_CASG_rbf = simplex_grad_mse(S_CASG_rbf, grad_est=grad_CASG_rbf, grad_true=grad_true, sig=sig)

    # CFD
    S_CFD = CFD.gen_sample_set(len(H), sig, h)
    grad_CFD = simplex_grad(F, x_0, S_CFD)
    loss_CFD = simplex_grad_mse(S_CFD, grad_est=grad_CFD, grad_true=F.f1(x_0), sig=sig)

    # RBF
    grad_RBF = F_RBF.f1(x_0)
    loss_RBF = np.linalg.norm(grad_RBF - grad_true)**2

    return {"FFD": loss_FFD, "CASG": loss_CASG, "CFD": loss_CFD, 
            "FFD_rbf": loss_FFD_rbf, "CASG_rbf": loss_CASG_rbf, "loss_RBF": loss_RBF}


def get_loss(F, sig, h, x_0):
    H = F.f2(x_0)
    Q = 1/2. * H
    S_FFD = FFD.gen_sample_set(Q, sig, h)
    S_CASG = eCASG.gen_sample_set(Q, sig, h)

    grad_FFD = simplex_grad(F, x_0, S_FFD)
    grad_CASG = simplex_grad(F, x_0, S_CASG)

    loss_FFD = simplex_grad_mse(S_FFD, grad_est=grad_FFD, grad_true=F.f1(x_0), sig=sig) 
    loss_CASG = simplex_grad_mse(S_CASG, grad_est=grad_CASG, grad_true=F.f1(x_0), sig=sig)

    return {"FFD": loss_FFD, "CASG": loss_CASG}

# ===== construct df =====
def construct_df(configs, res_list):
    rows = []
    for i in range(len(res_list)):
        conf = configs[i]
        dict_list = {}
        for r in res_list[i]:
            for method in r:
                if method not in dict_list:
                    dict_list[method] = []
                dict_list[method].append(r[method])
        
        # get statistics on loss
        for method in dict_list:
            curr_row = conf.copy()
            curr_row['method'] = method
            curr_row['median'] = np.median(dict_list[method])
            curr_row['25pct'] = np.percentile(dict_list[method], 25)
            curr_row['75pct'] = np.percentile(dict_list[method], 75)
            rows.append(curr_row)

    return pd.DataFrame(rows)

# ===== pre-compute points =====
def pre_compute_pts(F_class, configs, N, seed=21):
    """
    Pre-compute all points and RBF models to avoid shared state in parallel workers.
    RBF models are pre-computed by reusing points from previous RBF models.
    
    Args:
        F_class: The function class to use (e.g., CellDivReduced, Ackley)
        configs: List of configuration dictionaries
        N: Number of points to sample
        seed: Random seed for reproducible results
    
    Returns:
        points_cache: Dictionary of pre-computed points
        rbf_cache: Dictionary of pre-computed RBF models
    """
    print("Pre-computing caches...")
    points_cache = {}
    rbf_cache = {}
    
    # Set seed for reproducible cache generation
    np.random.seed(seed)
    print(f"Using seed: {seed}")
        
    # Get unique combinations of (d, sig) and (d, sig, N_RBF) and sort lexicographically
    unique_d_sig = sorted(set((conf['d'], conf['sig']) for conf in configs))
    unique_d_sig_nrbf = sorted(set((conf['d'], conf['sig'], conf['N_RBF']) for conf in configs))
    
    # Pre-compute points for each (d, sig) combination
    for d, sig in unique_d_sig:
        F = F_class(d=d)
        x_key = (d, sig)
        if x_key not in points_cache:
            xs = F.sample_domain_points(N, d)
            points_cache[x_key] = xs
            print(f"Pre-computed points for d={d}, sig={sig}")
    
    # Pre-compute RBF models for each (d, sig, N_RBF) combination
    # TODO: maybe have the sample points be noisy for RBF. Maybe not 
    # a good assumption that they are noise free.
    prev_d = None
    prev_sig = None
    prev_points = None
    prev_N = 0
    
    for d, sig, N_RBF in unique_d_sig_nrbf:
        rbf_key = (d, sig, N_RBF)

        if d != prev_d or sig != prev_sig:
            # New d or sig combination, start fresh
            prev_d = d
            prev_sig = sig
            prev_points = F.sample_domain_points(N_RBF, d)
            prev_N = N_RBF
            F = F_class(d=d)
        else:
            # Need more points, add the difference
            additional_points = F.sample_domain_points(N_RBF - prev_N, d)
            prev_points = np.vstack([prev_points, additional_points])
            prev_N = N_RBF

        # Use the accumulated points for RBF
        xs_rbf = prev_points
        F_RBF = RBF(xs_rbf, F.f_batch(xs_rbf), 0)
        rbf_cache[rbf_key] = F_RBF
        print(f"Pre-computed RBF for d={d}, sig={sig}, N_RBF={N_RBF}")
    
    print("Caches pre-computed successfully!")
    return points_cache, rbf_cache

# ==== plotting =====
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D

# Global configuration
mpl.rcParams.update({
    "text.usetex": True,                  # Use LaTeX if True (requires setup)
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",                # or 'sans-serif'
    "font.serif": ["Computer Modern Roman"],     # Preferred font
    "font.size": 12,                       # Base font size
    "axes.labelsize": 12,                  # Axis label font size
    "axes.titlesize": 11,                  # Title font size
    "legend.fontsize": 10,                  # Legend font size
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1,                   # Thickness of axes lines
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,                # Line thickness
    "lines.markersize": 5,                 # Marker size
    "figure.dpi": 150,                     # Resolution for saving
    "savefig.dpi": 150,
    "figure.figsize": [3.4, 2.4],          # Size in inches (IEEE column width)
    "legend.frameon": True,               # No box around legend
})

mpl.rcParams.update({
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "axes.grid.which": "both",  # Show grid on both major and minor ticks
})


figure_methods_confs = {
    'CFD': {
        'color': 'navy',
        'marker': 'd',
        'linestyle': '-',
        'label': 'CD',
        'markersize': 6
    },
    r'CASG': {
        'color': 'darkorange',
        'marker': 'o',
        'linestyle': '-',
        'label': r'CASG',
        'markersize': 6
    },
    r'CASG_rbf': {
        'color': 'darkorange',
        'marker': 'o',
        'linestyle': '--',
        'label': r'CASG+GM',
        'markersize': 6
    },
    r'FFD': {
        'color': 'slategray',
        'marker': 'x',
        'linestyle': '-',
        'label': r'FFD',
        'markersize': 6
    },
    r'FFD_rbf': {
        'color': 'slategray',
        'marker': 'x',
        'linestyle': '--',
        'label': r'FFD+GM',
        'markersize': 6
    },
    'loss_RBF': {
        'color': 'maroon',
        'marker': 's',
        'linestyle': '-',
        'label': 'RBF',
        'markersize': 5
    }
}

def _plot_figs_N_RBF(curr_dict, numerics_path=None, exp_name=None, plot_name=None):
    handles = []
    
    # Plot order: CASG first, then others for better visibility
    method_order = ['CASG', 'CASG_rbf', 'FFD', 'FFD_rbf', 'CFD', 'loss_RBF']
    
    for method in method_order:
        if method not in curr_dict:
            continue
            
        curr_method_df = curr_dict[method]
        x = curr_method_df['N_RBF']
        y = curr_method_df['median']
        y_low = curr_method_df['25pct']
        y_high = curr_method_df['75pct']

        style = figure_methods_confs[method]

        # Plot central line with higher zorder for CASG methods
        zorder = 10 if 'CASG' in method else 1
        line, = plt.plot(x, y, linestyle=style['linestyle'], color=style['color'],
                marker=style['marker'], label=style['label'], markersize=style['markersize'],
                zorder=zorder)

        # Plot very transparent error band to reduce clutter
        plt.fill_between(x, y_low, y_high, color=style['color'], alpha=0.15, zorder=zorder-1)
        
        # Store handle for legend (dot + line)
        handles.append(Line2D([0], [0],
                              linestyle=style['linestyle'],
                              color=style['color'],
                              marker=style['marker'],
                              label=style['label'],
                              markersize=style['markersize'],
                              linewidth=line.get_linewidth()))

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r"$N_{RBF}$")
    plt.ylabel(r"MSE")

    if plot_name is not None:
        # Save main plot without legend
        plt.savefig(f"{numerics_path}/figures/{exp_name}/{plot_name}.pdf", bbox_inches='tight')

        # Create shared legend figure
        fig_legend = plt.figure(figsize=(4, 0.4))
        ax = fig_legend.add_subplot(111)
        ax.axis("off")
        ax.legend(handles=handles, loc='center', ncol=len(handles), frameon=False)

        legend_path = f"{numerics_path}/figures/{exp_name}/{plot_name}_legend.pdf"
        fig_legend.savefig(legend_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_legend)

    plt.show()

def plot_figs_N_RBF(exp_name, df, numerics_path):
    keys = ['method', 'sig', 'd', 'N_RBF']

    # Group by 'group' and get index of rows with minimum 'b' in each group
    min_h_idx = df.groupby(keys)['median'].idxmin()

    # Use these indices to get the desired rows
    result = df.loc[min_h_idx].reset_index(drop=True)

    grouped = result.groupby(['sig', 'd'])

    figure_data = {}
    for key, group_df in grouped:
        figure_data[key] = {}
        for k_method, group_df_method in group_df.groupby('method'):
            figure_data[key][k_method] = group_df_method.sort_values("N_RBF")

    os.makedirs(f"{numerics_path}/figures/{exp_name}", exist_ok=True)
    for key in figure_data:
        plot_name = f"sig_{key[0]}_d_{key[1]}"
        print(plot_name)
        _plot_figs_N_RBF(figure_data[key], numerics_path, exp_name, plot_name)


def plot_figs_exact_loss(df, main_config, exp_name, numerics_path):
    os.makedirs(f"{numerics_path}/figures/{exp_name}", exist_ok=True)

    x = main_config['kappa']
    for d in main_config['d']:

        # FFD Plot
        ffd_df = df.query(f"d == {d} and method == 'FFD'")
        style = figure_methods_confs['FFD']
        # Plot central line
        plt.plot(x, ffd_df['median'], linestyle=style['linestyle'], color=style['color'],
                marker=style['marker'], label=style['label'], markersize=style['markersize'])

        # Plot filled error band
        plt.fill_between(x, ffd_df['25pct'], ffd_df['75pct'], color=style['color'], alpha=0.3)    

        # CASG Plot
        casg_df = df.query(f"d == {d} and method == 'CASG'")
        style = figure_methods_confs['CASG']
        # Plot central line
        plt.plot(x, casg_df['median'], linestyle=style['linestyle'], color=style['color'],
                marker=style['marker'], label=style['label'], markersize=style['markersize'])

        # Plot filled error band
        plt.fill_between(x, casg_df['25pct'], casg_df['75pct'], color=style['color'], alpha=0.3)    

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(r"$\kappa$")
        plt.ylabel("MSE")

        plt.legend()

        plot_name = f"d_{d}"

        plt.savefig(f"{numerics_path}/figures/{exp_name}/{plot_name}.pdf", bbox_inches='tight')
        plt.show()
