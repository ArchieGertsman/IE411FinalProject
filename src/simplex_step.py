import numpy as np



def simplex_step(A, B_inv, b_bar, c, iB, rule):
    cB      = c[:,iB]
    w       = cB @ B_inv
    z       = cB @ b_bar

    neg_reduced_costs = w@A - c
    best_cost_idx, best_cost = \
        select_pivot(neg_reduced_costs, rule)

    if best_cost <= 0:
        # at optimal solution
        return (-1, iB, b_bar, B_inv, z)

    y = B_inv @ A[:, best_cost_idx]
    indices = np.argwhere(y > 0).reshape(-1)
    if len(indices) == 0:
        # unbounded
        return (16, iB, b_bar, B_inv, np.inf)

    ratios = b_bar[indices] / y[indices]
    min_ratio_idx = indices[ratios.argmin()]

    b_bar = b_bar[:,None]
    y = y[:,None]
    z = z[:,None]
    best_cost = best_cost[:,None]
    tableau = np.vstack((
        np.hstack([  w  ,   z  , best_cost]),
        np.hstack([B_inv, b_bar,     y    ])
    ))
    tableau = pivot(tableau, min_ratio_idx+1)
    
    iB[min_ratio_idx] = best_cost_idx
    b_bar = tableau[1:,-2]
    b_bar = np.squeeze(np.asarray(b_bar))
    B_inv = np.asarray(tableau[1:, :b_bar.size])
    z = tableau[0, w.size]

    return (0, iB, b_bar, B_inv, z)



def pivot(tableau, idx):
    # create a 1 in the last column of the pivot row
    tableau[idx] /= tableau[idx, -1]

    multipliers = tableau[:,-1].reshape(-1,1).copy()

    # so that pivot row is not modified in next operation
    multipliers[idx] = 0.

    # create zeros in the last column of all the other rows
    tableau -= multipliers * tableau[idx]
    return tableau



def select_pivot(neg_reduced_costs, rule):
    if rule == 0:
        # Dantzig's rule
        c_best_idx = neg_reduced_costs.argmax()
        c_best = neg_reduced_costs[:,c_best_idx]
    else:
        # Bland's rule
        indices = np.argwhere(neg_reduced_costs > 0)[:,-1]
        if len(indices) > 0:
            c_best_idx = indices[0]
            c_best = neg_reduced_costs[:,c_best_idx]
        else:
            c_best_idx = None
            c_best = 0

    return (c_best_idx, c_best)




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

    print(simplex_step(A, B_inv, b_bar, c, iB, rule=0))