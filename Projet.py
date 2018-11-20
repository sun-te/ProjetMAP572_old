
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
        #direction of the edge: from point V to its precedor
        self.edge=np.zeros((n,1))
        self.pos=np.array([[np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
        
        matrice=self.adjacence
        matrice[0,0]=1;
        degree=np.array([1])
        
        
        '''
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
        '''
        #Variante
        #Si le degree est plus grand que 2:
        #P'-P
        # x= degree
        #=\frac{(2k-1)(x+\delta) -x(2k-1+k\delta)}{(2k-1+k\delta) \times (2k-1) }
        #=\frac{((2-x)k-1)\}{(2k-1+k\delta) \times (2k-1) }
        # if delta <0, on a renforc\'e 贫富差距
        # if delta >0, we have reduced the 贫富差距
        delta=10.
        for i in range(1,self.n):
            proba=np.cumsum((degree+delta)/(2*i-1+i*delta))
            flag=np.random.uniform(0,1)
            for j in range(len(proba)):
                if(flag<=proba[j]):
                    v=j
                    break;
            self.edge[v]=i;
            matrice[i,v]=1;
            matrice[v,i]=1;
            degree=np.sum(matrice,axis=0)
        self.adjacence=matrice 
        tmp=self.Distance()
        self.distance=tmp
        
       
    def Adjacence(self):

        return self.adjacence;
    
    def RenomalisedAdjacence(self):
        matrix=self.adjacence.copy()
        degree=np.sum(matrix,0)
        for i in range(len(matrix)):
            matrix[i,:]/=degree[i];
        return matrix
    
    def Distance_physque(self):
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
    def Distance(self):
        n=self.n
        A=self.adjacence.copy()
        tmp=self.adjacence.copy()
        matrix_dis=tmp.copy()
        dis=1
        matrix_dis[0,0]=0
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
        temp=(B-M)/M
        
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
                print("iteration, residu, d(步长): ",iteration, residu,d)
                
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
    def VisualCluster(self,dict_cluster_index):
        plt.figure(figsize=[8,8])
        pos = self.pos#np.array([[np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
        edge = []
        for i in range(self.n):
            for j in range(i,self.n):
                if (self.adjacence[i,j] == 1):
                    edge.append(([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]]))
        for i in range(len(edge)):
            plt.plot(*edge[i],color='blue')
            
        colors=['red','blue','green','yellow','purple','brown','black']
        i=0
        
    
        for key in dict_cluster_index.keys():
            
            for p in dict_cluster_index[key]:
                plt.scatter(self.pos[p][0],test.pos[p][1],color=colors[i])
            i+=1;
        plt.scatter(pos.T[0],pos.T[1])


#%%
'''
tool function
'''
def k_means(k,data,iter_max=10000):
    n,p=data.shape
    choix=np.random.randint(low=0,high=n,size=k)
    center0=[]
    
    for i in range(k):
        
        center0+=[data[choix[i]]]
    
    it=0
    
    while  it<=iter_max:
        #err_last=err_now
        err_now=0
        it+=1
        dict_clus={}
        dict_clus_index={}
        for i in range(k):
            dict_clus[i]=[]
            dict_clus_index[i]=[]
            
        for i in range(n):
            cluster=-1
            min_dis=np.inf
            for j in range(len(center0)):
                dis=np.linalg.norm(data[i]-center0[j])
                if(dis<min_dis):
                    min_dis=dis
                    cluster=j
            err_now+=min_dis
            
            dict_clus_index[cluster].append(i)
            dict_clus[cluster].append(data[i])
        center1=[]
        for key in dict_clus:
            
            len_cluster=len(dict_clus[key])
            if(len_cluster==0):
                continue
            center1+=[sum(dict_clus[key])/len_cluster]
        #print(len(center0))
        if(len(center0)==len(center1) and np.linalg.norm(np.array(center0)-np.array(center1))==0.0):
            print("OK")
            break;
        #print(center0)
        center0=center1.copy()
        
        
        if(len(center0)!=k):
            choix=np.random.randint(low=0,high=n,size=k)
            center0=[]
            
            for i in range(k):
                
                center0+=[data[choix[i]]]
                
        print(len(center0))
        

        
           
    return dict_clus,dict_clus_index     
def UnnormSpectralCluster( k,graph):
    W=graph.adjacence
    n=len(W)
    s=np.sum(W,0)
    D=np.zeros((n,n))
    for i in range(n):
        D[i,i]=s[i]
    L=D-W
    temp=np.linalg.eig(L)
    vector= np.real(temp[1])
    U=vector[:,0:k]
    dict_cluster_y, dict_cluster_index=k_means(k,U)
    for k in dict_cluster_index.keys():
        print(len(dict_cluster_index[k]))
    colors=['red','blue','green','brown','yellow','purple','black']
    i=0
    plt.figure(figsize=[10,10])

    for key in dict_cluster_index.keys():
        
        for p in dict_cluster_index[key]:
            plt.scatter(graph.pos[p][0],test.pos[p][1],color=colors[i])
        i+=1;

    return;

    
    