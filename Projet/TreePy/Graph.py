#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
       Graphe Structure Definition ans Visualization 
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
import matplotlib.pyplot as plt

class grapheG:
    def __init__(self,n=1):
        self.n=n
        self.adjacence=np.zeros((n,n))   
        #self.edge=np.zeros((n,1))
        self.pos=np.array([[np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
        self.distance=np.zeros((n,n))

    def reload(self,A):
        self.__init__(len(A))
        self.adjacence=A
        self.distance = self.Distance()
        '''for i in range(self.n):
            for j in range(self.n):
                if A[i,j]>0:
                    v = j
                    break
            self.edge[v] = i;
        '''

    def Distance(self):
        n=self.n
        A=self.adjacence.copy()
        tmp=self.adjacence.copy()
        matrix_dis=tmp.copy()
        dis=1
        for i in range(n):
            matrix_dis[i,i]=0
        for it in range(n):
            dis+=1
            tmp=np.dot(tmp,A)
            flag=1
            for i in range(n):
                for j in range(n):
                    if(i!=j and matrix_dis[i,j]==0 and tmp[i,j]!=0):
                        matrix_dis[i,j]=dis
                        flag=0
            if(flag==1):
                break;        
        return matrix_dis

    def Energy(self,p):
        n=self.n
        matrix_distance=self.distance
        max_d=np.max(matrix_distance)
        matrix_distance=matrix_distance/max_d
        for i in range(n):
            matrix_distance[i,i]=1
        x=np.array([p[:,0]])
        Xminus=(x.T-x)**2
        y=np.array([p[:,1]])
        Yminus=(y.T-y)**2
        B=np.sqrt(Xminus+Yminus)/np.sqrt(2)
        M=matrix_distance
        temp=(B-M)[np.where(M!=0)]/M[np.where(M!=0)]
        E=np.sum(temp*temp)
        return E

    def Gradiant(self,pos,epsilon=1.e-4):
        l=len(pos)
        res=np.zeros([l,2])
        E0=self.Energy(pos)
        for i in range(l):
            increment=np.zeros([l,2])
            increment[i,0]=epsilon
            res[i,0]=(self.Energy(pos+increment)-E0)/epsilon
            increment[i,0]=0
            increment[i,1]=epsilon
            res[i,1]=(self.Energy(pos+increment)-E0)/epsilon
        #res=
        return res 

    def GPF(self,delta=1.e-2,itermax=1000,tol=0.1):
        pos0=self.pos
        residu=np.inf
        iteration=0
        while (iteration<itermax and residu>tol):
            iteration+=1
            pos1=pos0-delta*(self.Gradiant(pos0))
            residu=self.Energy(pos1)
            pos0=pos1
            if(iteration%10==0):
                print(iteration, residu)
                self.pos=pos1
                self.Visual()
        self.pos=pos1
        self.Visual()


    def Gradiant_1d(self,x,epsilon=1.e-4):
        n=self.n
        l=n*2
        grad=np.zeros(l)
        
        increment=np.zeros(l)
        E0=self.Energy(x.reshape((n,2)))
        for i in range(l):
            increment[i]=epsilon
            grad[i]=(self.Energy((x+increment).reshape((n,2)))-E0)/epsilon
            increment[i]=0
        return grad

    def GPO_Armijio(self,itermax=1000,tol=1.e-4):
        n=self.n
        iteration=0
        residu=np.inf
        u=self.pos.reshape(n*2)
        while (iteration<itermax and residu>tol):
            w=self.Gradiant_1d(u)
            residu=w.dot(w)
            E0=self.Energy(u.reshape((n,2)))            
            rho=1
            while self.Energy((u-rho*w).reshape((n,2)))>E0-rho*0.0001*residu:#armijio critère, recherche linéaire
                rho*=0.8
            u-=rho*w
            iteration+=1
            if(iteration%5==0):
                print("iteration, residu, d(步长): ",iteration, residu,rho)
                #if(iteration%200==0):
                self.pos=u.reshape((n,2))
                self.Visual(draw_edge=False)
        print(iteration,residu,rho)        
        self.pos=u.reshape((n,2))
        self.Visual(draw_edge=False)
        
    def Visual(self,draw_edge=True):
        plt.figure(figsize=[6,6])
        pos = self.pos
        edge = set()
        for i in range(self.n):
            for j in range(self.n):
                if (i!=j and self.adjacence[i,j] >1.e-5):
                    x1,x2=pos[i][0],pos[j][0]
                    y1,y2=[pos[i][1],pos[j][1]]
                    if(x1>x2):
                        x1,x2=x2,x1
                        y1,y2=y2,y1
                    edge.add(((x1,x2),(y1,y2)))
        edge=list(edge)
        if(draw_edge):
            for i in range(len(edge)):
                plt.plot(*edge[i])
            for i in range(self.n):
                plt.annotate(s=i ,xy=(pos[i]))
        plt.scatter(pos.T[0],pos.T[1])
        #
        plt.show()

    def VisualCluster(self,cluster):
        plt.figure(figsize=[8,8])
        pos = self.pos
        colors=['red','blue','green','brown','purple','black','yellow','orange','pink']
        for i in range(len(pos)):    
            plt.scatter(pos[i][0],pos[i][1],color=colors[int(cluster[i])])
            

    def GPO_d(self,itermax=1000,tol=1.e-4):
        x0=self.pos.reshape(self.n*2)
        residu=np.inf
        iteration=0
        d=1.e-4
        grad0=self.Gradiant_1d(x0)
        E0=self.Energy(self.pos)
        while (iteration<itermax and residu>tol):
            iteration+=1
            x1=x0-d*(grad0)
            grad1=self.Gradiant_1d(x1)
            E1=self.Energy(x1.reshape((self.n, 2)))
            residu=np.abs(E1-E0)
            dp=(x1-x0)
            dD=(grad1-grad0)
            d=np.dot(dp,dD)/(np.linalg.norm(dD))**2
            grad0=grad1
            x0=x1
            E0=E1
            if(iteration%5==0):
                print("iteration, residu, d(步长): ",iteration, residu,d)
                #if(iteration%200==0):
                self.pos=x1.reshape((self.n, 2))
                self.Visual(draw_edge=False)
        print(iteration, residu,d)        
        self.pos=x1.reshape((self.n, 2))
        self.Visual(draw_edge=False)
