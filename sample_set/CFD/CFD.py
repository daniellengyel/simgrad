import numpy as np

def gen_sample_set(dim, sig, h=10, min_length=1e-10):
    return h * np.hstack([np.eye(dim), -np.eye(dim)])