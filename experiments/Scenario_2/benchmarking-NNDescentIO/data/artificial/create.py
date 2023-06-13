#Importar as bibliotecas necessárias
import numpy as np
from time import time   
import pandas as pd
from time import sleep



np.random.seed(0)



def set_colors(rows,N):
    colors = np.zeros(N)
    
    cores = rows.shape[0]
    sample = rows.shape[1]
    
    for k in range(cores):
        for i in range(sample):
            if rows[k,i]:
                colors[i] = k + 1
    return colors

#Concatena as distribuições probabilísticas
def make_sample(data):
    
    
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample,data[i]))
    
    
    return sample

def make_colors(colors):
    
    sample = colors[0]
    max_c = max(colors[0])
    
    for i in range(1,len(colors)):
        colors[i] = colors[i] + max_c + 1   
        max_c = max(colors[i])
        sample = np.concatenate((sample,colors[i]))
    return sample

def biclust_dataset(N,dim):
    #Building make_bicluster dataset
    from sklearn.datasets import make_biclusters
    X0, rows,_ = make_biclusters(
    shape=(N, dim), n_clusters=2, noise=.4,minval=-12,maxval=10, shuffle=False, random_state=10)
    y0 = set_colors(rows,N) #Colors
    
    return X0,y0

def blobs_dataset(N,dim):
    #Building make_blobs dataset
    from sklearn.datasets import make_blobs
    X1, y1 = make_blobs(n_samples=N, centers=5, n_features=dim,
                   random_state=10,cluster_std=.6)
    return X1,y1

def normalize_dataset(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data)
    
    return norm_data

def get_artifical_db(N,dim):
    
    old_n = N
    N = N//2
    
    x0,y0 = biclust_dataset(N,dim)
    
    x1,y1 = blobs_dataset(N,dim)
    
    data = [x0,x1]
    colors = [y0,y1]
    
    sample = make_sample(data)
    col_list = make_colors(colors)
    
    #É preciso normalizar o conjunto de dados, visto que a distância utilizada é a euclidiana
    np.random.shuffle(sample)
    return sample,col_list

def create_dataset(N,dim):
    
    sample,col_list = get_artifical_db(N,dim)
    colors = col_list
    N = sample.shape[0]
    i0 = 0
    for i in range(N//2,len(colors),N):
        
        c_unique = colors[i0:i]
        c_out = colors[i:]
        

      
    return sample.astype(np.float32),col_list



import sys


N = int(sys.argv[1])


db = create_dataset(int(N),12)[0]


path_file  = f'/nndescent/GPU_KNNG/data/artificial/SK_data.txt'

np.savetxt(path_file, db, delimiter='\t',fmt='%.30f',header=f'{N} 12',comments='')

