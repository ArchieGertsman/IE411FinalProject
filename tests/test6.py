# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:02:17 2017

@author: Siqi Miao
"""

# test6.py
#
# First initialization test: infeasible problem.
#
# indices of iB, iN start with 1

import sys
sys.path.append('../src/')

import numpy as np
from simplex_init import simplex_init
    
A = np.array([
   [1, 1, 1, 2,  1, 3],
   [1, 1, 0, 2,  2, 2],
   [1, 0, 0, 12, 1, 1]],
   dtype=np.float64)
b = np.array([-1,3,-1], dtype=np.float64)
c = np.array([[-1,-1,-1,-1,-1,-1]], dtype=np.float64)

status, _, _, _, _, _ = simplex_init(A, b, rule=0)

if (status != 16):
   print('istauts WRONG!!!!\n')

A = -A
status, _, _, _, _, _ = simplex_init(A, b, rule=0)

if (status != 16):
   print('istauts WRONG!!!!\n');



