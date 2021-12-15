# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:39:46 2017

@author: Siqi Miao
"""
# test5.py
#
# Unboundedness test.
#
# indices of iB, iN start with 1

import sys
sys.path.append('../src/')

import numpy as np
from simplex_step import simplex_step

# start with a tableau form
A = np.array([
   [-1,    1,     2],
   [-1,    1,     1],   
   [ 0,    1,     1]],
   dtype=np.float64)   
A = np.hstack((np.eye(3), A))
b = np.array([1,2,3],dtype = np.float64)
iB = np.arange(3)
xB = b.copy()
c  = np.array([[0,0,0,-1,2,1]], dtype=np.float64)

# form an invertible matrix B and modify the problem
B = np.array([
   [4, 1, 0],
   [1, -2, -1],
   [1, 2, 4]],
   dtype=np.float64)
B_inv = np.linalg.inv(B)
A = B @ A
b = B @ b

# modify c
iN = np.arange(A.shape[1])
iN = iN[np.isin(iN, iB, invert=True)]
N = A[:,iN]
c1 = np.array([[1, 1, 0]], dtype=np.float64)
c2 = c[:, (4-1):6] + (c1 @ B_inv @ N)
c = np.hstack((c1,c2))

B_inv = np.linalg.inv(A[:,iB])

# take a step.
status, iB, xB, _, _ = \
   simplex_step(A, B_inv, xB, c, iB, rule=0)


X = np.zeros(6, dtype=np.float64)
X[iB] = xB

if (status != 16):
   print('INCORRECT ISTATUS!\n')

