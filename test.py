# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:50:21 2018

@author: TeTe
"""

import scipy as sp
from scipy.optimize import  minimize
from Projet import grapheG

#%%
def Gradient_py(graph):
    x0=graph.Two2One(graph.pos)
    residu=np.inf
    iteration=0
    d=1.e-4
    grad0=graph.Gradiant_1d(x0)
    E0=graph.Energy(graph.pos)
    
    minimize()
            
#%%    
N =100

test=grapheG(N)

test.Visual()
a=test.Distance()
test.Energy(test.pos)
test.GPO(tol=1.e-3)
#UnnormSpectralCluster(4,test)

#%%
'''Part 3'''
MatriceAdjacence=np.loadtxt('StochasticBlockModel.txt')

#%%
'''Part 4'''

def Scores(A,eps,tol=1.e-6):
    n=len(A)
    Pe=(1-eps)*A+np.ones([n,n])*eps/n
    tup=np.linalg.eig(Pe)
    values=np.real(tup[0])
    vectors=np.real(tup[1].T)
    for i in range(len(values)):
        if(np.abs(values[i]-1)<=tol):
            index=i;
    
    return vectors[index]

A=test.RenomalisedAdjacence()
#for eps in [0,1.e-3, 1.e-2, 1.e-1,5.e-1]:
    #print(Scores(A,eps))
    