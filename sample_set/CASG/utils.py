import numpy as np
from numba import njit

@njit
def spectral_norm(S):
    """Compute the spectral norm (2-norm)."""
    u, s, vh = np.linalg.svd(S)
    return s[0]

@njit
def is_invertible(S, eps=1e-10):
    """Check invertibility via condition number."""
    cond = np.linalg.cond(S)
    return cond < 1 / eps

@njit
def loss_S(S, H, sigma, h):
    """
    Compute ell_{H, sigma}(S) = AE + NE if S is invertible and ||S||_2 <= h
    - H: symmetric (d, d)
    - sigma, h: scalars > 0
    - S: (d, d) matrix
    """
    d = S.shape[0]

    if not is_invertible(S):
        return np.inf

    if spectral_norm(S) > h + 1e-8:
        return np.inf

    # Compute inverse and transpose
    S_inv = np.linalg.inv(S)
    S_inv_T = S_inv.T

    # p_H(S): each p_i = s_iᵀ H s_i
    p = np.empty(d)
    for i in range(d):
        s = S[:, i]
        p[i] = s.T @ H @ s

    # Approximation Error
    AE = np.sum((S_inv_T @ p)**2)

    # Noise Error
    one_vec = np.ones(d)
    NE = sigma**2 * (np.sum(S_inv**2) + np.sum((S_inv_T @ one_vec)**2))
    return AE + NE
