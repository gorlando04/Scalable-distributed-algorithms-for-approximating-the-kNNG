name = '/nndescent'
try:
	import faiss
	name = '/rapids'
except:
	pass


import sys

N = int(sys.argv[1])


file = open(f"{name}/GPU_KNNG/results/sift1m_knng_k30.kgraph.txt", "r")

a = file.readlines()

import numpy as np

#N = int(1e6)
K = 32
I = np.zeros((N,K))

aux = [[] for x in range(0,N)]


for index,i in enumerate(a):


    valores = i.split("\t")
    
    for index2,b in enumerate(valores):
        
        if '\n' in b:
            b = b.split('\n')[0]
        if len(b) != 0 and index2 != 0:
            aux[index].append(int(b))


    I[index] = aux[index]

file.close()
print(I.shape)





file = open(f"{name}/GPU_KNNG/brute/SK-{N//int(1e6)}M_gt.txt", "r")

a = file.readlines()




#N = int(1e6)
I_gt = np.zeros((N,20))

aux = [[] for x in range(0,N)]


for index,i in enumerate(a):

    valores = i.split(" ")
    
    for index2,b in enumerate(valores):
        
        if '\n' in b:
            b = b.split('\n')[0]
        if len(b) != 0:
            aux[index].append(int(b))

    I_gt[index] = aux[index]

file.close()

print(I_gt.shape)


def rec_k(arr1,arr2,k):

    recall_value = (arr1[:,:k] == arr2[:,:k]).sum() / (float(N*k))

    print(f"Recall@{k} = {recall_value}")

print("Antes de tudo vamos verificar:")
print((I == 0).sum())
print((I_gt == 0).sum())

rec_k(I,I_gt,5)

rec_k(I,I_gt,10)

rec_k(I,I_gt,15)

rec_k(I,I_gt,20)


i = int(input("Digite um id: "))

while True:
    try:
        print(f"Vetor {i} aproximado: ",end=' ')
        for j in range(12):
            print(f"{I[i][j]}",end=' ')
        print(f"\n\nVetor {i} exato: ",end=' ')
        for j in range(12):
            print(f"{I_gt[i][j]}",end=' ')
    except:
        print("Nao existe esse ID")
    print("\n")
    i = int(input("Digite um id: "))
