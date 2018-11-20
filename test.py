# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:50:21 2018

@author: TeTe
"""

import scipy as sp
from scipy.optimize import  minimize
from Graphe import grapheG
from Part3 import Unnormalized,Normalized_sym
import numpy as np
#%%    
N =100

graph=grapheG()
graph.Initial_with_n(N,delta=10)

graph.Visual()
a=graph.Distance()
graph.Energy(graph.pos)
graph.GPO(tol=1.e-3)
#%%
cluster_un=Unnormalized(5,graph)
cluster_n=Normalized_sym(5,graph)
#%%
'''Part 3'''
MatriceAdjacence=np.loadtxt('StochasticBlockModel.txt')

graph1=grapheG()
graph1.Initial_with_A(MatriceAdjacence)
graph1.Visual()



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
    