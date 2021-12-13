from simplex_method import simplex_method
import numpy as np

def simplex_init(A, b, rule):
    m,n = A.shape
    A = np.hstack((A, np.eye(m)))
    c = np.hstack((np.zeros(n), np.ones(m)))
    B_inv = np.eye(m)
    iB = n + np.arange(m)

    status, x, z, iB, b_bar = \
        simplex_method(A, b, c, rule, B_inv, iB)

    if iB[iB >= n].size > 0:
        # infeasible
        return (16, iB, np.zeros(m))


if __name__ == "__main__":
    pass