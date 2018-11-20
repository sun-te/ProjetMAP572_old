def Distance(self):
        n=self.n
        pos=self.pos
        matrix_ref=np.zeros([n,n])
        matrix_adj=self.adjacence.copy()
        matrix_adj[0,0]=0
        for i in range(n):
            for j in range(n):
                if(matrix_adj[i,j]!=0):
                    matrix_ref[i,j]=np.linalg.norm(pos[i]-pos[j])
                else:
                    matrix_ref[i,j]=np.inf
        matrix_dis=matrix_ref
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    matrix_dis[i,j]=min(matrix_dis[i,j],matrix_dis[i,k]+matrix_dis[k,j])
        return matrix_dis
        
        
