import numpy as np

import sample_set.CASG.CASG as CASG

def sub_sample_set(idxs, S_out, D, sig, h, min_length=1e-10):
    if len(idxs) == 0:
        return S_out
    D_curr = D[idxs]
    S_curr = CASG.gen_sample_set(np.diag(D_curr), sig, h, min_length)
    for i in range(len(idxs)):
        for j in range(len(idxs)):
            S_out[idxs[i], idxs[j]] = S_curr[i, j]

    return S_out

def gen_sample_set(H, sig, h, min_length=1e-10):
    d = H.shape[0]

    # we get the difference vectors in the rotated space
    D, R = np.linalg.eigh(H)

    S = np.zeros(shape=(d, d))

    # The decomposition into d = 4 * k + 2 * l + 1 * m,
    # where l, m < 2.
    k = d // 4
    l = (d - k * 4) // 2
    m = d % 2 
    all_idxs = list(range(d))
    # idxs_4 = all_idxs[:k*2] + all_idxs[d - k*2:]
    # idxs_2 = all_idxs[k*2:k*2 + l] + all_idxs[d - k*2 - l:d - k*2]
    # idxs_1 = all_idxs[k*2 + l:k*2 + l + m]


    idxs_4 = all_idxs[:k*4]
    idxs_2 = all_idxs[k*4:k*4 + 2*l]
    idxs_1 = all_idxs[k*4 + 2*l:k*4 + 2*l + m]

    S = sub_sample_set(idxs_4, S, D, sig, h, min_length)
    S = sub_sample_set(idxs_2, S, D, sig, h, min_length)
    S = sub_sample_set(idxs_1, S, D, sig, h, min_length)

    return R @ S


if __name__ == "__main__":
    d = 5
    H = np.random.normal(size=(d, d))
    H = (H.T + H)/2.

    sig = 0.1
    h = 0.1
    print(gen_sample_set(H, sig, h))