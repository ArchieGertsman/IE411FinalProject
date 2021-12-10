import numpy as np

def simplex_step(A, b, c, iB, iN, xB, B_inv, rule):
    cB      = c[:,iB]
    w       = cB @ B_inv
    b_bar   = B_inv @ b
    z       = cB @ b_bar

    tableau = np.vstack((
        np.hstack((w,     z    )),
        np.hstack((B_inv, b_bar))
    ))

    neg_reduced_costs = w@A - c

    if rule == 0:
        # Dantzig's rule
        c_best_idx = neg_reduced_costs.argmax()
        c_best = neg_reduced_costs[:,c_best_idx]
    else:
        # Bland's rule
        indices = np.argwhere(neg_reduced_costs > 0)
        if len(indices) > 0:
            c_best_idx = indices[0]
            c_best = neg_reduced_costs[:,c_best_idx]
        else:
            c_best = 0

    if c_best <= 0:
        # print("done")
        return (-1, iB, iN, xB, B_inv)

    y = B_inv @ A[:, c_best_idx]
    indices = np.argwhere(y > 0)
    if len(indices) == 0:
        # print("unbounded")
        return (16, iB, iN, xB, B_inv)

    ratios = b_bar[indices] / y[indices]
    min_ratio_idx = indices[ratios.argmin()]

    tableau = np.hstack((tableau, np.vstack(c_best, y)))
    
    # TODO: pivot at tableau[-1, min_ratio_idx]
    # TODO: update iB, iN, xB, B_inv

    return (0, iB, iN, xB, B_inv)



if __name__ == "__main__":
    A1 = np.matrix([[1, 1,  1],
                [1, 1, -1],
                [1, 1,  0]],dtype = np.float64)

    A = np.hstack((np.eye(3),A1))

    b = np.matrix([[1],
                [2],
                [3]],dtype = np.float64)
                

    iB = [1,2,3]
    iN = [4,5,6]
    xB = np.matrix(np.copy(b))
    c  = np.matrix([[0,0,0,-1,2,1]],dtype = np.float64)

    B_inv = np.linalg.inv(A[:,iB])

    # test a step in this extremely simple state
    irule = 0
    print(simplex_step(A,b,c,iB,iN,xB, B_inv, irule))