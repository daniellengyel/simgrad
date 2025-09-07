import os
import pickle

def load_hadamard(d):
    assert (d <= 2 or d % 4 == 0) and (d <= 246)
    base_dir = os.path.dirname(__file__)  # directory of the current script
    filepath = os.path.join(base_dir, 'data', f'had_{d}.pkl')
    
    with open(filepath, 'rb') as f:
        had = pickle.load(f)
    return had

