#Importar as bibliotecas necessárias
import faiss
import numpy as np
from time import time   
import pandas as pd
from time import sleep


ngpu = 3

## This is the replication method
class MultiGPUIndex:

    #Construtor da classe
    def __init__(self, data, name):
        
        self.name = name
        self.data = data

    #Destrutor da classe
    def __del__(self):
        del self.data

################################################################################################################
#                                                                                                              #
#                                           REPLICATION BF                                                     #     
#                                                                                                              #         
#                           This is the class that runs the replication BF                                     #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

class MultiBrute(MultiGPUIndex):

    def __init__(self, data, name):
        super().__init__(data, name)

        ## Multi-GPU config
        self.gpus = list(range(ngpu))
        self.res = [faiss.StandardGpuResources() for _ in self.gpus]
        self.co = faiss.GpuMultipleClonerOptions()


        
    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        ## Shape of the data
        n, d = self.data.shape

        ## Make the index on CPU
        index_cpu = faiss.IndexFlatL2(d)

        ## Make the index on Multi-GPU
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)

        

        ## Add the data to the index to have better performance
        index.add(self.data)

        t0 = 0
        ## Search
        _, I = index.search(self.data, k)
        
        tf = 0
        return I.astype(np.float32),tf


np.random.seed(0)

import sys


N = int(sys.argv[1])

file = open(f"/rapids/GPU_KNNG/data/artificial/SK-{N//int(1e6)}M_data.txt", "r")
#file = open(f"SK-{N//int(1e6)}M_data.txt", "r")

a = file.readlines()


db = np.zeros((N,12))

aux = [[] for x in range(0,N)]


for index,i in enumerate(a):

    if index != 0:
        valores = i.split("\t")
        
        for index2,b in enumerate(valores):
            
            if '\n' in b:
                b = b.split('\n')[0]
            if len(b) != 0:
                aux[index-1].append(float(b))


        db[index-1] = aux[index-1]

file.close()


db = db.astype(np.float32)



def bvecs_read(fname):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy()

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#Reading vector
#SIFT = fvecs_read('/rapids/GPU_KNNG/data/sift/sift_base.fvecs')

#db = SIFT

I,t = MultiBrute(db,'brute').search(21)
I = I[:,1:]
print(I.shape)



np.savetxt(f'SK-{N//int(1e6)}M_gt.txt', I, delimiter=' ',fmt='%.0f')


