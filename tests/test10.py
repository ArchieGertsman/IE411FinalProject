# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:02:17 2017

@author: Siqi Miao
"""

# test10.py
#
# An infeasible input to simplex method.
#
# indices of iB, iN start with 1

import sys
sys.path.append('../src/')

import numpy as np
from simplex_method import simplex_method


# first form an invertible matrix
R = np.array([[
    4, 1,  1],
   [ 1, 2,  1],
   [ 1, 1,  1]],
   dtype=np.float64)

# form a vector b which is in the span of R
b = R @ np.array([1,-4,-1] ,dtype=np.float64)

B = np.array([
   [1, 1, 1],
   [1, 1, 0],
   [1, 0, 0]],
   dtype=np.float64)
A = np.hstack((R,B))

c = np.array([[-2, 1, 1, -1, -1, -1]], dtype=np.float64)

status, _, _, _, _, _ = \
   simplex_method(A, b, c, rule=1)

if (status != 4):
   print('istatus is wrong\n');
