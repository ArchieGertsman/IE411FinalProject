# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:39:46 2017

@author: Siqi Miao
"""

# test3.py
#
# First pivot test.
#
# indices of iB, iN start with 1

import sys
sys.path.append('../src/')

import numpy as np
from numpy.linalg import norm
from simplex_step import simplex_step

# start with a tableau form
A = np.array([
   [1, 1, 2],
   [1, 1, 1],
   [1, 1, 1]],
   dtype=np.float64)
A = np.hstack((np.eye(3), A))
b = np.array([1,2,3], dtype=np.float64)
iB = np.arange(3)
xB = b.copy()
c  = np.array([[0,0,0,-1,-2,1]], dtype=np.float64)

# form an invertible matrix B and modify the problem
B = np.array([
   [4, 1, 0],
   [1, -2, -1],
   [1, 2, 4]],
   dtype=np.float64)
A = B @ A
b = B @ b

B_inv = np.linalg.inv(A[:,iB])

# take a step with the first rule
status, iB, xB, _, _ = \
   simplex_step(A, B_inv, xB, c, iB, rule=0)


X = np.zeros(6, dtype=np.float64)
X[iB] = xB

if (status != 0):
   print('INCORRECT ISTATUS!\n')
   
if (norm(X-np.array([0,1,2,0,1,0]) > 1e-10)):
   print('INCORRECT STEP!\n')

if (norm(np.array(sorted(iB))-np.array([1,2,4])) > 1e-10):
   print('iB incorrect!\n')






