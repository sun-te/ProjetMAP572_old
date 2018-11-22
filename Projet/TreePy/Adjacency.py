#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
       Random Graphe Matrix Generation 
'''

__author__ = "Te SUN et Enlin ZHU"
__copyright__ = "Copyright 2018, MAP572"
__credits__ = ["Te SUN", "Enlin ZHU", "Prof",]
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Te SUN"
__email__ = "te.sun@polytechnique.edu"
__status__ = "Test"

import numpy as np

def matrix_delta(n,delta=0):
    n=int(n)
    ans = np.zeros((n,n))
    ans[0,0] = 1;
    degree = np.array([1])
    for i in range(1,n):
        proba=np.cumsum((degree+delta)/(2*i-1+i*delta))
        flag=np.random.uniform(0,1)
        for j in range(len(proba)):
            if(flag<=proba[j]):
                v=j
                break
        ans[i,v],ans[v,i] = 1,1
        degree=np.sum(ans,axis=0)
    return ans
  
def matrix_SBM(partition,Q):
    n = len(partition)
    ans = np.zeros((n,n))
    for i in range(n):
        r = int(partition[i])
        ans[i,i] = 1
        for j in range(i):
            flag=np.random.uniform()
            s=int(partition[j])
            if(flag<Q[s,r]):
                ans[i,j],ans[j,i] = 1,1
    return ans

def RenomalizedMatrix(A):
    degree=np.sum(A,0)
    for i in range(len(A)):
        A[i,:]/=degree[i];
        



