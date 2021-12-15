# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:02:17 2017

@author: Siqi Miao
"""
# test9.py
#
# Solve a simplex method problem test.
#
# indices of iB, iN start with 1

import sys
sys.path.append('../src/')

import numpy as np
import random
from simplex_method import simplex_method
from simplex_test import simplex_test


# first form an invertible matrix
R = np.array([
   [ 4, 1,  1],
   [-1, 2,  1],
   [ 1, 1, -1]],
   dtype=np.float64)

# form a vector b which is in the span of R
b = R @ np.array([1,2,1], dtype=np.float64)

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

c = np.array([[-2, 1, 1, -1, -1, -1]], dtype=np.float64)
c = c[:,p]

# test
status, x, z, iB, xB, B_inv = \
   simplex_method(A, b, c, rule=0)

if (status != 0) :
   print('istatus is wrong\n')

_, _, isfeasible, isoptimal, _ = \
   simplex_test(A, b, c, iB, xB)

if (isfeasible != 1):
   print('your solution is not feasible!!!\n')

if (isoptimal != 1):
   print('your solution is not optimal!!!\n')

