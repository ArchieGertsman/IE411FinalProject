from simplex_step import simplex_step
import simplex_init
import numpy as np


def simplex_method(A, b, c, rule, B_inv=None, iB=None):
    if B_inv is None:
        status, A, b, B_inv, iB, b_bar = \
            simplex_init.simplex_init(A, b, rule)
        if status == 4:
            return (16,) + (None,)*5 # failed
        elif status == 16:
            return (4,) + (None,)*5 # infeasible
    
    status = 0
    b_bar = B_inv@b
    while status == 0:
        status, iB, b_bar, B_inv, z = \
            simplex_step(A, B_inv, b_bar, c, iB, rule)
    
    x = np.zeros(A.shape[1])
    x[iB] = b_bar.reshape(-1)
    if status == -1:
        return (0, x, z, iB, b_bar, B_inv) # success
    elif status == 16:
        return (32,) + (None,)*5 # unbounded
