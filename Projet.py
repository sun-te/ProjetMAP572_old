
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%

class grapheG:
    def __init__(self,n):
        self.n=n
        self.adjacence=np.zeros((n,n))       
        self.pos=np.array([[np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
        matrice=self.adjacence
        matrice[0,0]=1;
        degree=np.array([1])
        for i in range(1,self.n):
            proba=np.cumsum(degree/(2*i-1))
            flag=np.random.uniform(0,1)
            for j in range(len(proba)):
                if(flag<=proba[j]):
                    v=j
                    break;
            matrice[i,v]=1;
            matrice[v,i]=1;
            degree=np.sum(matrice,axis=0)
        self.adjacence=matrice 
        
        
    def Adjacence(self):

        return self.adjacence;
    
   

    def Distance(self):
        n=self.n
        pos=self.pos
        matrix_ref=np.zeros([n,n])
        matrix_adj=self.adjacence.copy()
        matrix_adj[0,0]=0
        for i in range(n):
            for j in range(n):
                if(matrix_adj[i,j]!=0 or i==j):
                    matrix_ref[i,j]=np.linalg.norm(pos[i]-pos[j])
                else:
                    matrix_ref[i,j]=np.inf
        matrix_dis=matrix_ref
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis=np.min(matrix_dis[i,:]+matrix_ref[:,j])
                    if(matrix_dis[i,j]>dis):
                        matrix_dis[i,j]=dis
        return matrix_dis;
    def Energy(self,p):
        n=self.n
        matrix_distance=self.Distance()
        E=0
        max_d=np.max(matrix_distance)
        for i in range(n):
            for j in range(n):
                E+=(np.linalg.norm(p[i]-p[j])/np.sqrt(2)-matrix_distance[i,j])**2
        E/=max_d**2
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
        
    def One2Two(self,x):
        l=int(len(x)/2)
        res=np.array([x[0:l],x[l:]]).T
        return res
    def Two2One(self,pos):
        return np.append(pos[:,0],pos[:,1])
    
    def Gradiant_1d(self,x,epsilon=1.e-4):
        l=len(x)
        grad=np.zeros(l)
        E0=self.Energy(self.One2Two(x))
        
        for i in range(l):
            increment=np.zeros(l)
            increment[i]=epsilon
            grad[i]=(self.Energy(self.One2Two(x+increment))-E0)/epsilon

        return grad
    def GPO(self,itermax=1000,tol=1.e-4):
        
        x0=self.Two2One(self.pos)
        residu=np.inf
        iteration=0
        d=1.e-4
        grad0=self.Gradiant_1d(x0)
        E0=self.Energy(self.pos)
        
        while (iteration<itermax and residu>tol):
            iteration+=1
            x1=x0-d*(grad0)
            grad1=self.Gradiant_1d(x1)
            E1=self.Energy(self.One2Two(x1))
            residu=np.abs(E1-E0)
            
            dp=(x1-x0)
            dD=(grad1-grad0)
            
            d=np.dot(dp,dD)/(np.linalg.norm(dD))**2
            
            grad0=grad1
            x0=x1
            E0=E1
            
            if(iteration%2==0):
                print(iteration, residu,d)
                
                #if(iteration%200==0):
                self.pos=self.One2Two(x1)
                self.Visual()
            
        print(iteration, residu,d)        
        self.pos=self.One2Two(x1)
        self.Visual()

    def Visual(self):
        plt.figure(figsize=[6,6])
        pos = self.pos#np.array([[np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
        edge = []
        for i in range(self.n):
            for j in range(i,self.n):
                if (self.adjacence[i,j] == 1):
                    edge.append(([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]]))
        for i in range(len(edge)):
            plt.plot(*edge[i])
        plt.scatter(pos.T[0],pos.T[1])
        for i in range(self.n):
            plt.annotate(s=i ,xy=(pos[i]))
        plt.show()
            
#%%    
N = 30
test=grapheG(N)

test.Visual()
a=test.Distance()
test.Energy(test.pos)
test.GPF()
#%%

#%%

        
    
    
    
    