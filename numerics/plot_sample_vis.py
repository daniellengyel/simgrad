
import numpy as np

from pathlib import Path
import os

import sample_set.CASG.eCASG as eCASG
import sample_set.FFD.FFD as FFD

from functions.quadratic import Quadratic
from numerics.utils import figure_methods_confs

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def ill_cond_operator(kappa, v):
    d = len(v)
    v = v / np.linalg.norm(v)
    return np.eye(d) + np.outer(v, v) * (kappa - 1)

def rotate_scale_Q(Q, kappa):
    # rotate matrix
    Q_rot = np.linalg.qr(np.random.randn(d, d))[0]
    Q = Q_rot.T @ Q @ Q_rot

    # Add ill conditioning
    v = np.random.normal(size=d)
    T = ill_cond_operator(kappa**0.5, v)
    Q = T.T @ Q @ T
    return Q


def plot_sample_set(Q, d, h, sig, fig_name):
    
    F = Quadratic(Q, b=np.zeros(d), c=0, sig=0)    

    S_FFD = FFD.gen_sample_set(Q, sig, h)
    S_CASG = eCASG.gen_sample_set(Q, sig, h)

    lim = max(np.max(np.abs(S_CASG)), np.max(np.abs(S_FFD)))*1.1


    X = np.linspace(-lim, lim, 100)
    X, Y = np.meshgrid(X, X)
    XY = np.vstack([X.ravel(), Y.ravel()])
    Z = F.f_batch(XY.T).reshape(X.shape[0], Y.shape[0])


    # Plot contours and labels
    cp = plt.contour(X, Y, Z, colors="black", vmax=1, linewidths=1, alpha=0.25) 
    cp.clabel()

    # Plot points
    plt.scatter(S_CASG[0], S_CASG[1], color=figure_methods_confs['CASG']['color'], zorder=3, s=50)
    plt.scatter(S_FFD[0], S_FFD[1], color=figure_methods_confs['FFD']['color'], marker="d", zorder=3, s=50)

    # Plot lines
    plt.plot([0, S_CASG[0, 0]], [0, S_CASG[1, 0]], linewidth=2, linestyle=figure_methods_confs['CASG']['linestyle'], color=figure_methods_confs['CASG']['color'], alpha=0.75, zorder=2)
    plt.plot([0, S_CASG[0, 1]], [0, S_CASG[1, 1]], linewidth=2, linestyle=figure_methods_confs['CASG']['linestyle'], color=figure_methods_confs['CASG']['color'], alpha=0.75, zorder=2)

    plt.plot([0, S_FFD[0, 0]], [0, S_FFD[1, 0]], linewidth=2, linestyle="--", color=figure_methods_confs['FFD']['color'], alpha=0.75, zorder=2)
    plt.plot([0, S_FFD[0, 1]], [0, S_FFD[1, 1]], linewidth=2, linestyle="--", color=figure_methods_confs['FFD']['color'], alpha=0.75, zorder=2)

    # Axes limits and labels
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    # --- Custom legend handles ---
    casg_handle = Line2D([0], [0], marker='o', color="darkorange", label="CASG",
                        markersize=6, linewidth=1)
    ffd_handle = Line2D([0], [0], marker='d', color="slategrey", label="FFD",
                        markersize=6, linewidth=1, linestyle='--')

    plt.legend(handles=[casg_handle, ffd_handle])

    figs_path = Path("numerics/figures/sample_vis")
    os.makedirs(figs_path, exist_ok=True)

    plt.savefig(figs_path / f"{fig_name}.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    d = 2
    h = 1
    sig = 1e-3

    np.random.seed(1)

    Q = np.diag([1, 5])
    plot_sample_set(Q, d, h, sig, fig_name='def1')

    Q = rotate_scale_Q(np.diag([1, 1]), 0.1)
    plot_sample_set(Q, d, h, sig, fig_name='def2')

    Q = np.diag([0.1, -1])
    plot_sample_set(Q, d, h, sig, fig_name='indef1')

    Q = rotate_scale_Q(np.diag([1, -1]), 0.01)
    plot_sample_set(Q, d, h, sig, fig_name='indef2')

