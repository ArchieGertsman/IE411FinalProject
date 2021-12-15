# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:02:17 2017

@author: Siqi Miao
"""

# test7.py
#
# Feasible initialization test.
#
# indices of iB, iN start with 1


import sys
sys.path.append('../src/')

import numpy as np
from numpy.random import randn
import random
from numpy.linalg import norm,cond
from simplex_init import simplex_init

eps=1.0e-10;

# first form an invertible matrix
R = np.array([
   [ 4, 1,  1],
   [-1, 2,  1],
   [ 1, 1, -1]],
   dtype=np.float64)

# form a vector b which is in the span of R
b = R @ abs(randn(3))

B = np.array([
   [1, 1, 1],
   [1, 1, 0],
   [1, 0, 0]],
   dtype=np.float64)
A = np.hstack((R,B))


# form a random permutation
p = list(range(0,6))
random.shuffle(p) 
A = A[:,p]

status, _, _, _, iB, xB = simplex_init(A, b, rule=0)

X = np.zeros(6, dtype=np.float64)
X[iB] = xB

if (status != 0): 
   print('istatus wrong!\n')

# test feasibility
if (norm(A@X-b) > eps):
   print('NOT FEASIBLE!!!\n')


if (min(X) < 0):
   print('NOT FEASIBLE!!!\n')

# test that we have a basis
if (1/cond(A[:,iB])) > 1.0e6:
   print('NOT BASIC!!!\n')