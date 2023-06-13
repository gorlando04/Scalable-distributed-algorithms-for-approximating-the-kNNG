name = '/nndescent'

import sys

where = ''

VAL = sys.argv[1]



N = int(VAL)


file = open(f"{name}/GPU_KNNG/results/NNDescent-KNNG.kgraph.txt", "r")

a = file.readlines()

import numpy as np

K = 32
I = np.zeros((N,K))

aux = [[] for x in range(0,N)]


for index,i in enumerate(a):

    if index == N:
        break

    valores = i.split("\t")
    
    for index2,b in enumerate(valores):
        
        if '\n' in b:
            b = b.split('\n')[0]
        if len(b) != 0 and index2 != 0:
            aux[index].append(int(b))


    I[index] = aux[index]

file.close()

print(I.shape)





file = open(f"{name}/GPU_KNNG/brute/SK-{int(VAL)/1e6}M_gt.txt", "r")

a = file.readlines()


I_gt = np.zeros((N,19))

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

assert I_gt.shape[0] == I.shape[0]



def rec_k(arr1,arr2,k):

    recall_value = (arr1[:,:k] == arr2[:,:k]).sum() / (float(N*k))

    print(f"Recall@{k} = {recall_value}")


print("Antes de tudo vamos verificar:")
print((I == 0).sum())
print((I_gt == 0).sum())

rec_k(I,I_gt,5)

rec_k(I,I_gt,10)

rec_k(I,I_gt,15)




