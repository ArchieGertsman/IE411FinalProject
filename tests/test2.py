# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:39:46 2017

@author: Siqi Miao
"""
# test2.py
#
# Test that the simplex method code takes a correct step
# when the form is slightly more complicated. 
#
# indices of iB, iN start with 1

import sys
sys.path.append('../src/')

import numpy as np
from numpy.linalg import norm
from simplex_step import simplex_step

A = np.array([
   [-4,    1,     0,    -3,    -3,    -5],
   [1,    -2,    -1,    -2,    -2,     3],
   [1,     2,     4,     7,     7,    -1]],
   dtype=np.float64)
b = np.array([-2,-6,17], dtype=np.float64)              
iB = np.arange(3)
B_inv = np.linalg.inv(A[:,iB])
xB = np.array([1,2,3], dtype=np.float64)
c  = np.array([[1,1,1,2,5,1]], dtype=np.float64)

# take a step
status, iB, xB, _, _ = \
   simplex_step(A, B_inv, xB, c, iB, rule=0)

X = np.zeros(6, dtype=np.float64)
X[iB] = xB

if (status != 0):
   print('INCORRECT ISTATUS!\n')
   
if (norm(X-np.array([0,1,2,1,0,0])) > 1e-10):
   print('INCORRECT STEP!\n')

if (norm(np.array(sorted(iB))-np.array([1,2,3])) > 1e-10):
   print('iB incorrect!\n')



