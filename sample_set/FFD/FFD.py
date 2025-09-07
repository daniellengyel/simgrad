import numpy as np

def gen_sample_set(H, sig, h=10, min_length=1e-10):
    """To not use H, set h and min_length to the same value."""
    H_diag = np.abs(np.diag(H))
    S = np.diag(np.where(H_diag != 0, 2 * np.sqrt(sig / np.abs(H_diag)), np.inf))

    # clamp difference vector lengths between h and min_length
    S = np.minimum(S, h)
    S = np.maximum(S, min_length)
    return S
    
