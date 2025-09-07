import numpy as np

from sample_set.CASG.hadamard.io import load_hadamard
from sample_set.CASG.gen_sing_vals.gen_sing_vals import solve_sing_vals_exact

def permute_rows(M, i, j):
    tmp_row = M[i].copy()
    M[i] = M[j].copy()
    M[j] = tmp_row
    return M

def gen_sample_set(H, sig, h, min_length=1e-9):
    d = H.shape[0]

    D_diag, R = np.linalg.eigh(H)

    V_T = load_hadamard(d)
    sing_vals = np.maximum(solve_sing_vals_exact(D_diag, sig, h), min_length)

    k = np.argmax(sing_vals)
    V_T = permute_rows(V_T, 0, k)

    sing_vals = np.diag(sing_vals)

    return R @ sing_vals @ V_T

if __name__ == "__main__":
    d = 8
    H = np.random.normal(size=(d, d))
    H = (H.T + H)/2.

    sig = 0.1
    h = 0.1
    print(gen_sample_set(H, sig, h))