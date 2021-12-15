# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:04:29 2017

@author: Siqi Miao
"""
# Test the feasibility and optimality of a basic vector for the
# linear program
#
# indices of iB, iN start with 1
import numpy as np
from numpy.linalg import norm

def simplex_test(A, b, c, iB, xB):
    n = A.shape[1]
    eps = 1.0e-12
    
    X = np.zeros(n)
    X[iB] = xB
    eta=c*X

    err = norm(A@X-b)
  
    isfeasible = 0
    if (err < eps) and min(X) >= -eps:
        isfeasible = 1
        
    iN = np.arange(A.shape[1])
    iN = iN[np.isin(iN, iB, invert=True)]
    
    Cb = c[:,iB]  
    Cn = c[:,iN] 
    B = A[:,iB]
    
    Binv = np.linalg.inv(B)
    
    ctilde = (Cn - Cb @ Binv @ A[:,iN]).reshape(-1)
    isoptimal = int(min(ctilde) >= -eps)
    return (X, eta, isfeasible, isoptimal, ctilde)