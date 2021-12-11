from simplex_step import simplex_step
from simplex_init import simplex_init
import numpy as np

def simplex_method(A, b, c, rule, B_inv=None, iB=None):
    if B_inv is None:
        status, iB, b_bar = simplex_init(A, b, c, rule)
        if status == 4:
            # failed
            return (16, np.zeros(A.shape[1], np.inf, iB, b_bar))
        elif status == 16:
            # infeasible
            return (4, np.zeros(A.shape[1], np.inf, iB, b_bar))
    
    status = 0
    b_bar = B_inv@b
    while status == 0:
        status, iB, b_bar, B_inv, z = \
            simplex_step(A, B_inv, b_bar, c, iB, rule)
    
    x = np.zeros(A.shape[1])
    x[iB] = b_bar.reshape(-1)
    if status == -1:
        return (0, x, z, iB, b_bar) # success
    elif status == 16:
        return (32, x, z, iB, b_bar) # unbounded

    


if __name__ == "__main__":
    A = np.array([
        [1, 1,  1],
        [1, 1, -1],
        [1, 1,  0]],
        dtype = np.float64)
    A = np.hstack((np.eye(3),A))
    B_inv = np.eye(3)
    b_bar = np.array([1,2,3], dtype=np.float64)
    c  = np.matrix([[0,0,0,-1,2,1]], dtype=np.float64)
    iB = [0,1,2]

    print(simplex_method(A, b_bar, c, rule=0, B_inv=B_inv, iB=iB))