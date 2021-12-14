from simplex_step import simplex_step
import simplex_init
import numpy as np


def simplex_method(A, b, c, rule, B_inv, iB):
    if B_inv is None:
        status, A, b, B_inv, iB, b_bar = \
            simplex_init.simplex_init(A, b, rule)
        if status == 4:
            return (16,) + (None)*5 # failed
        elif status == 16:
            return (4,) + (None)*5 # infeasible
    
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
        return (32,) + (None)*5 # unbounded

    


if __name__ == "__main__":
    # A = np.array([
    #     [1, 1,  1],
    #     [1, 1, -1],
    #     [1, 1,  0]],
    #     dtype = np.float64)
    # A = np.hstack((np.eye(3),A))
    # B_inv = np.eye(3)
    # b_bar = np.array([1,2,3], dtype=np.float64)
    # c  = np.matrix([[0,0,0,-1,2,1]], dtype=np.float64)
    # iB = [0,1,2]

    # print(simplex_method(A, b_bar, c, rule=0, B_inv=None, iB=None))
    A = np.array([
        [1,1,1,0],
        [-1,1,2,0],
        [0,2,3,0],
        [0,0,1,1]
    ])
    c = np.array([[-1,2,-3,0]])
    b = np.array([6,4,10,2])
    print(simplex_method(A, b, c, rule=0, B_inv=None, iB=None))
