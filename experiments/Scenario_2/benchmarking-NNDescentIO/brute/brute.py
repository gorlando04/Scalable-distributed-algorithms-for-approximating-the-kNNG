## Importing required libs

import numpy as np
import time
import sys
import faiss
from multiprocessing.pool import ThreadPool
import pandas as pd
import gc
import math



ngpu = 3

BRUTE_SIZE = int(1e4)

NORM = False

np.random.seed(0)


################################################################################################################
#                                                                                                              #
#                                           DATASET FUNCTIONS                                                  #     
#                                                                                                              #         
#                   The functions below are related with the creationg of the artificial                       # 
#                 dataset that are going to be used in this benchmarking                                       #
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################




def set_colors(rows,N):
    colors = np.zeros(N)
    
    cores = rows.shape[0]
    sample = rows.shape[1]
    
    for k in range(cores):
        for i in range(sample):
            if rows[k,i]:
                colors[i] = k + 1
    return colors


## Join the arrays that have differente probabilistic distributions
def join_sample(data):
    
    
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample,data[i]))
    
    return sample

## Create the colors for each probabilistic distribution (importante for the future)
def make_colors(colors):
    
    sample = colors[0]
    max_c = max(colors[0])
    
    for i in range(1,len(colors)):
        colors[i] = colors[i] + max_c + 1   
        max_c = max(colors[i])
        sample = np.concatenate((sample,colors[i]))
    return sample

## Create a dataset with bicluster distributions
def biclust_dataset(N,dim):
    #Building make_bicluster dataset
    from sklearn.datasets import make_biclusters
    X0, rows,_ = make_biclusters(
    shape=(N, dim), n_clusters=2, noise=.4,minval=-12,maxval=10, shuffle=False, random_state=10)
    y0 = set_colors(rows,N) #Colors
    
    return X0,y0

## Create dataset with make_blobs distribution
def blobs_dataset(N,dim):
    #Building make_blobs dataset
    from sklearn.datasets import make_blobs
    X1, y1 = make_blobs(n_samples=N, centers=5, n_features=dim,
                   random_state=10,cluster_std=.6)
    return X1,y1

## Normalize the data, only if necessary
def normalize_dataset(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data)
    
    return norm_data

## Get the datasets with the propreties that is especified, and call the make col func
def get_artifical_db(N,dim):
    
    old_n = N
    N = N//2
    
    x0,y0 = biclust_dataset(N,dim)
    
    x1,y1 = blobs_dataset(N,dim)
    
    data = [x0,x1]
    colors = [y0,y1]
    
    sample = join_sample(data)
    col_list = make_colors(colors)
    
    normalized_sample= None
    if NORM:
        #É preciso normalizar o conjunto de dados, visto que a distância utilizada é a euclidiana
        normalized_sample = normalize_dataset(sample)
    else:
        normalized_sample = sample
    
    np.random.shuffle(normalized_sample)
    return normalized_sample,col_list

## Create the dataset by calling the functions above and check their integrity
def create_dataset(N,dim):
    
    sample,col_list = get_artifical_db(N,dim)
    colors = col_list
    N = sample.shape[0]
    i0 = 0
    for i in range(N//2,len(colors),N):
        
        c_unique = colors[i0:i]
        c_out = colors[i:]
        
        unique = np.sort(pd.unique(c_unique))
        unique_out = np.sort(pd.unique(c_out))
        
        i0 = i
        
        for i in unique:
            if i in unique_out:
                print(f"O valor {i} esta na lista {unique_out}")
                exit()
      
    return sample.astype(np.float32),col_list



################################################################################################################
#                                                                                                              #
#                                           GPU RESOURCES                                                      #     
#                                                                                                              #         
#                   In this area, all the functions related to create/set GPU                                  # 
#                 resources, as tempMem or the standart resources.                                             #
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################


def tempMem(tempmem):
    gpu_resources = []

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    return gpu_resources


################################################################################################################
#                                                                                                              #
#                                           DATA PROCESSING                                                    #     
#                                                                                                              #         
#                   This area is reserved for functions/classes that turns easier                              # 
#                 the management of the dataset and theirs properties                                          #
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################



class IdentPreproc:
    """
        a pre-processor is either a faiss.VectorTransform or an IndentPreproc
    """

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x

def get_preprocessor(xb):

    d = xb.shape[1]
    preproc = IdentPreproc(d)

    return preproc


## Transform the array to np.float32
def sanitize(x):
    """ 
        convert array to a c-contiguous float array because
        in faiss only np.float32 arrays can be processed in python
    """

    return np.ascontiguousarray(x.astype('float32'))

################################################################################################################
#                                                                                                              #
#                                           THREADING/ITERATORS                                                #     
#                                                                                                              #         
#                   This area is reserved for functions that control the thread pool                           #
#                    and the batch iteration of datasets.                                                      #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################


def rate_limited_imap(f, l):
    """
        A threaded imap that does not produce elements faster than they
        are consumed, and this is done to control the batches that are being 
        processed by the GPU
    """

    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


################################################################################################################
#                                                                                                              #
#                                           SHARDING BF                                                        #     
#                                                                                                              #         
#                           This is the class that runs the sharding BF                                        #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

class ShardedMultiGPUIndex:

    #Construtor da classe
    def __init__(self, data, name,gpu_resources):
        
        self.name = name
        self.data = data
        self.gpu_res = gpu_resources
        self.N,self.D = self.data.shape

    #Destrutor da classe
    def __del__(self):
        del self.data

    
    def make_vres_vdev(self,i0=0, i1=-1):
        "return vectors of device ids and resources useful for gpu_multiple"
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if i1 == -1:
            i1 = ngpu
        for i in range(i0, i1):
            vdev.push_back(i)
            vres.push_back(self.gpu_res[i])
        return vres, vdev

    def dataset_iterator(self,x, preproc, bs):
        """ 
            Set the begining and the end of each batch, this is 
            done by getting the batch_size and the number of samples
            and diviig one by another, to check how many batches will be 
            created. eg: nb = 100, bs = 10: The batches will be:

                [
                    (0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), 
                (70, 80), (80, 90), (90, 100)
                
                ]
        """

        nb = x.shape[0]
        block_ranges = [(i0, min(nb, i0 + bs))
                        for i0 in range(0, nb, bs)]

        ## This function makes sure that the array creates are np.float32
        def prepare_block(i01):
            i0, i1 = i01
            xb = sanitize(x[i0:i1])
            return i0, preproc.apply_py(xb)

        ## Return with help of the thread pool to speed up the transformation
        return rate_limited_imap(prepare_block, block_ranges)



        
class ShardingBrute(ShardedMultiGPUIndex):

    def __init__(self, data, name, gpu_resources,end=False):
        super().__init__(data, name, gpu_resources)
        self.end = end

    
    ## This is the sharding method
    def search(self,K):

        ## Initiate timer
        t0 = time.time()

        ## N_sample


        nq_gt = BRUTE_SIZE

        ## Instanciate the Distances and Indices arrays
        gt_I = np.zeros((nq_gt, K), dtype='int64')
        gt_D = np.zeros((nq_gt, K), dtype='float32')

        ## Using faiss heap to mantain the results ordered
        heaps = faiss.float_maxheap_array_t()
        heaps.k = K
        heaps.nh = nq_gt
        heaps.val = faiss.swig_ptr(gt_D)
        heaps.ids = faiss.swig_ptr(gt_I)
        heaps.heapify()


        ## Search batch size
        bs = int(16384*3)



        ## Make sure that this is the database to search
        xqs = self.data[:nq_gt]

        if self.end:
            xqs = self.data[-nq_gt:]

        ## Create the index
        db_gt = faiss.IndexFlatL2(self.D)
        vres, vdev = self.make_vres_vdev()

        ##Turn the index to Multi-GPU
        db_gt_gpu = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, db_gt)

        # compute ground-truth by blocks of bs, and add to heaps
        for i0, xsl in self.dataset_iterator(self.data, IdentPreproc(self.D), bs):
            db_gt_gpu.add(xsl)
            D, I = db_gt_gpu.search(xqs, K)
            I += i0
            heaps.addn_with_ids(
                K, faiss.swig_ptr(D), faiss.swig_ptr(I), K)
            db_gt_gpu.reset()
            print(f"\r{i0}/{self.N} ({(time.time()-t0):.3f} s) - brute" , end=' ')

        heaps.reorder()
        t1 = time.time()


        return gt_I,(t1-t0)



################################################################################################################
#                                                                                                              #
#                                           MAIN                                                               #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

import sys

def instanciate_dataset(n,d):

    start = 'SK-'


    number = str( int( n / 1e6) )
    base_name = start + number+ 'M-'

    name = base_name + str(d) + 'd'

    db = create_dataset(int(n),d)[0]
    
    return db, name




N = int(sys.argv[1])

db,_ = instanciate_dataset(N,12) 

print(f"Vamos inciar. N_SAMPLE = {db.shape[0]} e DIM = {db.shape[1]}")


gpu_resources = tempMem(-1)
index = ShardingBrute(db,'brute-begin',gpu_resources)

I,t = ShardingBrute(db,'brute-begin',gpu_resources).search(21)
I_gt_begin = I[:,1:]

print(f"O array de begin tem as seguintes dimensões: N_SAMPLE = {I_gt_begin.shape[0]} e DIM = {I_gt_begin.shape[1]}")

I,t = ShardingBrute(db,'brute-begin',gpu_resources,True).search(21)
I_gt_final = I[:,1:]

print(f"O array de end tem as seguintes dimensões: N_SAMPLE = {I_gt_begin.shape[0]} e DIM = {I_gt_begin.shape[1]}")

del db


print("Escrevendo os valores iniciais\n")
np.savetxt(f'SK-{N/1e6}M_gt-BEGIN.txt', I_gt_begin, delimiter=' ',fmt='%.0f')

print("Escrevendo os valores finais\n")
np.savetxt(f'SK-{N/1e6}M_gt-FINAL.txt', I_gt_final, delimiter=' ',fmt='%.0f')
