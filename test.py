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
'''Part 3'''
'''Question 3.1'''
N =100

graph=grapheG()
graph.Initial_with_n(N,delta=10)

graph.Visual()
a=graph.Distance()
graph.Energy(graph.pos)
graph.GPO(tol=1.e-3)
#%%
k=8
cluster_un=Unnormalized(k,graph)
cluster_n=Normalized_sym(k,graph)
#%%

'''Question 3.2'''
MatriceAdjacence=np.loadtxt('StochasticBlockModel.txt')

graph=grapheG()
graph.Initial_with_A(MatriceAdjacence)
graph.Visual()
#%%
graph.GPO(tol=0.01)
#%%
k=3
cluster_n=Normalized_sym(k,graph)



#%%

n=200
K=3
sommets=np.array(range(n))
partition=np.zeros(n)

for i in range(n):
    partition[i]=np.random.randint(0,K)

tmp=np.random.uniform(0,0.5,[K,K])
Q=tmp+tmp.T
Q=np.array([[0.8,0.02,0.2],[0.02,0.8,0.1],[0.2,0.1,0.8]])
eps=0.01
Q=np.ones([K,K])*eps
Q=Q+np.eye(K)*(1-eps)
#Q=np.array([[0.5,0.02,0.1,],[0.02,1]])
print(Q)
graph=grapheG()
graph.Initial_StochasticBlockModel(partition,Q)

#%%
graph.GPO()
graph.VisualCluster(partition)
#%%
cluster_n=Normalized_sym(3,graph)

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
    