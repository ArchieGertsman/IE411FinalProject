import simplex_method
from simplex_step import pivot
import numpy as np

def simplex_init(A, b, rule):
    m,n = A.shape
    A1 = np.hstack((A, np.eye(m)))
    c = np.hstack((np.zeros(n), np.ones(m)))
    c = c.reshape((1,) + c.shape)
    B_inv = np.eye(m)
    iB = n + np.arange(m)

    # perform the simplex method on the phase I LP
    # containing artificial variables
    _, x, _, iB, b_bar, B_inv = \
        simplex_method.simplex_method(A1, b, c, rule, B_inv, iB)

    # indices of artificial basic variables
    iB_artificial = iB[iB >= n]

    if iB_artificial.size == 0:
        # proceed to phase II
        return (0, A, b, B_inv, iB, b_bar)
    
    iB_artificial_values = x[iB_artificial]
    if iB_artificial_values[iB_artificial_values > 0].size > 0:
        # infeasible
        return (16,) + (None,)*5

    # artificial variables are still in the basis
    # at zero level
    return (0,) + __remove_artificial_variables(A, B_inv, b, iB, b_bar)
    


def __remove_artificial_variables(A, B_inv, b, iB, b_bar):
    n = A.shape[1]

    # construct tableau
    tableau = np.hstack((B_inv, b_bar[:,None]))
    # stack empty column to right of tableau
    tableau = np.hstack((
        tableau, np.zeros(tableau.shape[0])[:,None]))

    iB_legit = iB[iB < n]
    iN_legit = np.arange(n)
    iN_legit = iN_legit[np.isin(iN_legit, iB_legit, invert=True)]

    for i in iN_legit:
        artificial_rows = iB[iB >= n] - n
        if artificial_rows.size == 0:
            break # finished removing all artificial variables

        y = B_inv @ A[:,i]

        # find all pivotable rows, i.e. all rows j s.t. 
        # j corresesponds to an artificial variable and y[j] != 0
        pivotable_rows = artificial_rows[y[artificial_rows] != 0]
        if pivotable_rows.size == 0:
            continue # nothing to pivot on
            
        # update rightmost column of tableau to y
        tableau[:,-1] = y

        # pivot on first pivotable row
        j = pivotable_rows[0]
        tableau = pivot(tableau, j)
        iB[j] = i

    tableau, A, b, iB = \
        __delete_redundant_rows(tableau, A, b, iB)
    
    b_bar = tableau[:,-2]
    B_inv = tableau[:, :b_bar.size]
    return (A, b, B_inv, iB, b_bar)


def __delete_redundant_rows(tableau, A, b, iB):
    n = A.shape[1]

    rows_to_delete = iB[iB >= n] - n
    if rows_to_delete.size == 0:
        # no redundant rows
        return (tableau, A, b, iB)

    A = np.delete(A, rows_to_delete, 0)
    b = np.delete(b, rows_to_delete, 0)
    tableau = np.delete(tableau, rows_to_delete, 0)

    artificial_indices = np.argwhere(iB >= n)
    tableau = np.delete(tableau, artificial_indices, 1)
    iB = np.delete(iB, artificial_indices, 0)
    return (tableau, A, b, iB)