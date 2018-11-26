#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    test
    Created on Tue Nov 20 14:50:21 2018
    @author: TeTe
'''

import numpy as np
import os
os.chdir("C:\\Users\\TeTe\\Desktop\\Projet")
#import scipy as sp
#from scipy.optimize import  minimize
from TreePy import Cluster,Graph,Adjacency,PageRank
from bokeh.io import  show
from bokeh.plotting import figure
from bokeh.layouts import column,row
#%%
'''
Part1
'''
n=5000
graph=Graph.grapheG()
graph.reload(Adjacency.matrix_delta(n,0))
Adj=graph.adjacence
degree=np.sum(Adj,1)
degree.sort()
degrees=np.unique(degree)
count_d=[]
print("Construction complete")
for i in degrees:
    c=0
    for m in range(len(degree)):
        if(degree[m]==i):
            c+=1
    count_d+=[c]
count_d=np.array(count_d)/n
c=1.5
fig=figure(x_axis_type="log",y_axis_type="log",plot_width=800,plot_height=800, y_range=[0.8/n,1],title='Recheche de alpha')
fig.line(degrees,degrees**(-2)/c, line_width=2, legend="slope -2", color="red", line_dash='dotted')
fig.line(degrees,degrees**(-2.2)/c, line_width=2, legend="slope -2.2", color="green", line_dash='dotted')
fig.line(degrees,degrees**(-2.5)/c, line_width=2, legend="slope -2.2", color="orange", line_dash='dotted')
fig.line(degrees,degrees**(-3)/c, line_width=2, legend="slope -3", color="blue", line_dash='dotted')
fig.x(degrees, count_d,line_width=5)
show(fig)
#%%

'''Part4'''
M = np.array([[0,  0, 0, 0, 1],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0,   1, 1, 0, 0],
             [0,   0, 1, 1, 0]])



M = np.array([[1,0,0,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0],
              [1,1,0,0,0,0,0,0,0,0,0],
              [0,1,0,1,0,1,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0]
            ]).T


graph=Graph.grapheG()
graph.reload(M)
graph.GPO_Armijio()
graph.Visual(draw_edge=True)
scores=PageRank.Scores(graph.adjacence,eps=0.15)
print(scores)
#%%   

'''Part 3'''
'''Question 3.1'''
N =100
graph=Graph.grapheG()
graph.reload(Adjacency.matrix_delta(N,1))
graph.Visual()
a=graph.Distance()
graph.Energy(graph.pos)
graph.GPO_d(itermax=800,tol=1.e-3)
#%%
graph.GPO_Armijio(itermax=800,tol=1.e-3)
#%%

#%%
k=4
cluster_un=Cluster.Unnormalized(k,graph)
cluster_n=Cluster.Normalized_sym(k,graph)
#%%
'''Question 3.2'''
MatriceAdjacence=np.loadtxt('StochasticBlockModel.txt')
graph=Graph.grapheG()
graph.reload(MatriceAdjacence)
graph.Visual()
#%%
graph.GPO(tol=0.01)
#%%
k=3
cluster_n=Cluster.Normalized_sym(k,graph)
#%%
n=20
K=3
sommets=np.array(range(n))
partition=np.zeros(n)
for i in range(n):
    partition[i]=np.random.randint(0,K)
tmp=np.random.uniform(0,0.5,[K,K])
Q=tmp+tmp.T
Q=np.array([[0.8,0.02,0.2],[0.02,0.8,0.1],[0.2,0.1,0.8]])
eps=0.5
Q=np.ones([K,K])*eps
Q=Q+np.eye(K)*(1-eps)
print(Q)
graph=Graph.grapheG()
graph.reload(Adjacency.matrix_SBM(partition,Q))
#%%
graph.GPO(itermax=200)
#%%
graph.VisualCluster(partition)
cluster_n=Cluster.Normalized_sym(3,graph)



    