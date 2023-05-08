import faiss
import numpy as np
from time import time   
import pandas as pd
from time import sleep


## Global variable
NORM = False
HNSW = False
ngpu = 3

#################################################################################

##                              DATASET FUNCTIONS
##
##          The functions above are related with the creation of the artificial 
##          datasets that are going to be used in the benchmarking

#################################################################################


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

#################################################################################

##                              MEASUREMENT FUNCTIONS
##
##          The functions above are related with the measurement of the kNN methods
##          that are being tested in this benchmark

#################################################################################


## Calculate the recall@k of the method
def recall(arr1,arr2,k):
    
    #Verificação da integridade
    if arr1.shape != arr2.shape:
        print("Impossível de fazer a avaliação, as arrays tem tamanho diferentes")
    elif arr1.shape[1] < k:
        print(f"Impossível de fazer o recall{k}, já que as array não tem {k} vizinhos")
    
    #Somatório dos k primeiros vizinhos positivos dividido por n*k
    acertos = 0
    
    n = arr1.shape[0]


    recall_value = (arr1[:,:k] == arr2[:,:k]).sum() / (float(n*k))
    
    return recall_value
    
## Calculate the mean of the kNN times of the methods
def analysis_runtime(values):
    
    values = np.array(values)

    mean = values.mean()
  
    
    
    return mean

## Transform the array to np.float32
def sanitize(x):
    """ 
        convert array to a c-contiguous float array because
        in faiss only np.float32 arrays can be processed in python
    """

    return np.ascontiguousarray(x.astype('float32'))


def tempMem(tempmem):
    gpu_resources = []

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    return gpu_resources


gpu_res = tempMem(-1)


def make_vres_vdev(i0=0, i1=-1):
        "return vectors of device ids and resources useful for gpu_multiple"
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if i1 == -1:
            i1 = ngpu
        for i in range(i0, i1):
            vdev.push_back(i)
            vres.push_back(gpu_res[i])
        return vres, vdev
    
def train_coarse_quantizer(x, k):


    ## Gets the dimension
    d = 12

    ## Prepare the quantizer
    clus = faiss.Clustering(d, k)

    ## Set the maximum points per centroid
    clus.max_points_per_centroid = int(1e7)

    ## Instantiate the timer


    x = sanitize(x)

    ## Set the GPU resources to train the quantizer
    vres, vdev = make_vres_vdev()

    ## Multi-GPU index
    index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, faiss.IndexFlatL2(d))

    ## Train the quantizer (k-means)
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    ## Return the centroids
    return centroids.reshape(k, d)

def prepare_coarse_quantizer(data,nlist):

    nt = max(int(1e6), 256 * nlist)

    ## Training the quantizer
    centroids = train_coarse_quantizer(data[:nt], nlist)
    


    ## Add to the Index the centroids (PQ Technique)
    coarse_quantizer = faiss.IndexFlatL2(12)
    coarse_quantizer.add(centroids)

    return coarse_quantizer


#################################################################################

##                              kNN SEARCHING Class
##
##          The Class above are related with the kNN methods
##          that are being tested in this benchmark

#################################################################################

class MultiGPUIndex:

    #Construtor da classe
    def __init__(self, data, name):
        
        self.name = name
        self.data = data

    #Destrutor da classe
    def __del__(self):
        del self.data

    



class MultiBrute(MultiGPUIndex):

    def __init__(self, data, name):
        super().__init__(data, name)

        ## Multi-GPU config
        self.gpus = list(range(ngpu))
        self.res = gpu_res
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

        t0 = time()
        ## Search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf

class MultiIVFFlat(MultiGPUIndex):

    def __init__(self, data, name,nprobe,nlist):
        super().__init__(data, name)

        ## Approximate method settings
        self.nprobe = nprobe
        self.nlist = nlist

        ## Multi-GPU config
        self.gpus = list(range(ngpu))
        self.res = gpu_res
        self.co = faiss.GpuMultipleClonerOptions()



    def __del__(self):
#        del self.quantizer
        return super().__del__()

    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        t0 = time()
        quantizer = prepare_coarse_quantizer(self.data,self.nlist)
        self.train_time = time() - t0
        ## Data shape
        n, d = self.data.shape

        ## Creating the Index on CPU
        index_cpu = faiss.IndexIVFFlat(quantizer,d,self.nlist,faiss.METRIC_L2)

        ## Setting nprobe to the index
        index_cpu.nprobe = self.nprobe


        ## Making the Multi-GPU index
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)


    
        t0 = time()
        n = int(1e6)
        ## Training the index
        index.train(self.data[:n])
        self.train_time += time() - t0

        t0 = time()
        ## Adding the data to the index
        index.add(self.data)
        self.add_time = time() - t0

        t0 = time()
        ## Perform the search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf


class MultiIVFPQ(MultiGPUIndex):

    def __init__(self, data, name,nprobe,nlist,M):
        super().__init__(data, name)

        ## Approximate method settings
        self.nprobe = nprobe
        self.nlist = nlist
        self.M = M

        ## Multi-GPU config
        self.gpus = list(range(ngpu))
        self.res = gpu_res
        self.co = faiss.GpuMultipleClonerOptions()



    def __del__(self):
        return super().__del__()

    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        t0 = time()
        quantizer = prepare_coarse_quantizer(self.data,self.nlist)
        self.train_time = time() - t0
        ## Data shape
        n, d = self.data.shape

        ## Making the index on CPU
        index_cpu = faiss.IndexIVFPQ(quantizer,d,self.nlist,self.M,8,faiss.METRIC_L2)


        ## Setting nprobe to the index
        index_cpu.nprobe = self.nprobe


        ## Making the Multi-GPU index
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)

        t0 = time()
        n = int(1e6)
        ## Training the index
        index.train(self.data[:n])
        self.train_time += time() - t0

        t0 = time()
        ## Adding the data to the index
        index.add(self.data)
        self.add_time = time() - t0

        t0 = time()
        ## Perform the search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf

class MultiIVFSQ(MultiGPUIndex):

    def __init__(self, data, name,nprobe,nlist):
        super().__init__(data, name)

        ## Approximate method settings
        self.nprobe = nprobe
        self.nlist = nlist
        self.qtype = faiss.ScalarQuantizer.QT_8bit

        ## Multi-GPU config
        self.gpus = list(range(ngpu))
        self.res = gpu_res
        self.co = faiss.GpuMultipleClonerOptions()
    


    def __del__(self):
#        del self.quantizer
        return super().__del__()




    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        t0 = time()
        quantizer = prepare_coarse_quantizer(self.data,self.nlist)
        self.train_time = time() - t0
        ## Data shape
        n, d = self.data.shape


        ## Making the index on CPU
        index_cpu = faiss.IndexIVFScalarQuantizer(quantizer,d,self.nlist,self.qtype,faiss.METRIC_L2)


        ## Setting nprobe to the index
        index_cpu.nprobe = self.nprobe


        ## Making the Multi-GPU index
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)


        
        t0 = time()
        n = int(1e6)
        ## Training the index
        index.train(self.data[:n])
        self.train_time += time() - t0

        t0 = time()
        ## Adding the data to the index
        index.add(self.data)
        self.add_time = time() - t0

        t0 = time()
        ## Perform the search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf

## Transform the array to np.float32
def sanitize(x):
    """ 
        convert array to a c-contiguous float array because
        in faiss only np.float32 arrays can be processed in python
    """

    return np.ascontiguousarray(x.astype('float32'))

from multiprocessing.pool import ThreadPool

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


################################################################################################################
#                                                                                                              #
#                                           SHARDING BF                                                        #     
#                                                                                                              #         
#                           This is the class that runs the sharding BF                                        #     
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
    
class ShardingBrute(ShardedMultiGPUIndex):

    def __init__(self, data, name, gpu_resources):
        super().__init__(data, name, gpu_resources)

    
    ## This is the sharding method
    def search(self,K):

        ## Initiate timer
        t0 = time()

        ## N_sample
        nq_gt = self.N

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
        bs = 16354


        ## Make sure that this is the database to search
        xqs = self.data

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
            print(f"\r{i0}/{self.N} ({(time()-t0):.3f} s) - brute" , end=' ')

        heaps.reorder()
        t1 = time()


        return gt_I,(t1-t0)
    
#################################################################################

##                              AUXILIAR FUNCTIONS
##
##          The function below are related with auxiliar methods

#################################################################################

def write_df(df,index,info):

    for i in info.keys():
        if i != 'gpu_res' and i != 'data':
            df.loc[index,i] = info[i]

    return


def instanciate_dataset(n,d):

    start = 'SK-'


    number = str( int( n / 1e6) )
    base_name = start + number+ 'M-'

    name = base_name + str(d) + 'd'

    db = create_dataset(int(n),d)[0]
    
    return db, name

def create_object(name,info):

    index = None

    ## Create brute force index
    if name == 'brute' and info['data'].shape[0] <= int(30e6):
        index = MultiBrute(info['data'],name)
        return index
    elif name == 'brute':
        gpu_resources = tempMem(-1)
        index = ShardingBrute(info['data'],name,gpu_resources)
        return index


    D = info['data'].shape[1]



    ## Create IVFFlat index
    if name == 'ivfflat':
        index = MultiIVFFlat(info['data'],name,info['nprobe'],info['nList'])

    ## Create IVFPQ index
    elif name == 'ivfpq':

        #M_list = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96] 
        M = D


        index = MultiIVFPQ(info['data'],name,info['nprobe'],info['nList'],M)
    
    ## Create IVFSQ index
    else:
        index = MultiIVFSQ(info['data'],name,info['nprobe'],info['nList'])
    
    return index

#################################################################################

##                              PRIMARY EXECUTION
##
##          Above the primary execution is happening

#################################################################################


## Methods
metodos = {'deterministico': ['brute'],
           'probabilistico':['ivfflat',
                            'ivfpq',
                            'ivfsq'
                            ]
}



# Define the datasets n sample
dbs = range(int(0),int(506e6)+1,int(1e7))
d = 12

import math
def main():

    ## Initializating the dataframe
    df_gpu = df_gpu = pd.DataFrame()

    ## Control variable
    indice = 0


    ## Setting K value
    k = 20

    ## Setting nlist

    ## Setting recall@K value
    rec_k = 10

    ## Iterate the datasets
    for n in dbs:
        if n == 0:
            n = int(1e6)
        #Create the dataset
        data,name = instanciate_dataset(n,d)

        ## Create the variable to save the exact result
        brute_indices = None

        size = data.nbytes / 1e9

        for c in metodos: #deterministico e probabilistico
            for method in metodos[c]: #brute, ivfflat,ivfpq,ivfsq
                
                ## Warm-UP GPU
                if indice == 0:
                    print("Warming up the GPU...")
                    
                    _,_ = MultiBrute(data[:int(1e6)],method).search(k)
                    print("GPU Ready to go...")
                    sleep(5)
                


                #List if the are repetitions

                """
                Here, all the possible configurations that the approximate methods can be executed will be tested, and their results will be inserted
                on the list above. It will be possible to order the results by time or recall
                """

                ## Setting nlists and nprobes values for the approximate method
                nlist = int( 2 ** ( 2 * round(math.log(n,10)) ) )
                nprobes = [5,10,25,50,75,100]

                ## Setting random values for brute method
                if method == 'brute':
                    nlist = 0
                    nprobes = [0]

                ## Info declaration
                info = {}

                ## Primary iteration
                for nprobe in nprobes:
                    
                    #List if the are repetitions
                    results = []

                    ## Writing control variable
                    indice += 1

                    ## Set the information about the methdo, if the brute method is performed nlist and nprobe will not be used
                    info['nprobe'] = nprobe
                    info['data'] = data
                    n_sample,dim = (n,d)

                    info['Name'] = name
                    info['Method'] = method
                    info['Dim'] = dim
                    info['N_sample'] = n_sample
                    

                    info['nList'] = nlist
                    info['DB Size (GB)'] = size

                    try:
                        ## Create the object by the name 
                        index = create_object(method,info)

                        ## Perform the search, saving the indices and the kNN time
                        indices,time_knn = index.search(k)
                        
                        rec_value = '-'

                        ## Save the exact result
                        if method == 'brute':
                            brute_indices = indices.copy()
                            s = f'Recall@{rec_k}'
                            info['Time kNN'] = time_knn
                            info[s] = rec_value
                            info['Total'] = time_knn

                        ## Calculate the results and ADD them to the list
                        if method != 'brute':
                            rec_value = recall(brute_indices,indices,rec_k)
 
                            s = f'Recall@{rec_k}'
                            info['Time kNN'] = time_knn
                            info[s] = rec_value
                            info['Train time'] = index.train_time
                            info['Add time'] = index.add_time
                            info['Total'] = index.train_time + index.add_time + time_knn                    

                        #Save the results in a dataframe
                        time_knn = info['Time kNN']
                        rec_value = info[f'Recall@{rec_k}']
                        nlist = info['nList']
                        nprobe = info['nprobe']

                        
                        write_df(df_gpu,indice,info)

                        #Show the informations to check how the test is going and in what stage the test is on
                        print(f"Iteration -> {indice} DB -> {name} Dim -> {dim} N -> {n_sample} Finished in {time_knn:.5} secs, method -> {method}, recall -> {rec_value}, nlist -> {nlist} nprobe -> {nprobe} TOTAL -> {info['Total']:.3f}")            
                    
                    except Exception as e:
                        info['Erro'] = str(e)
                        write_df(df_gpu,indice,info)

                        #Show the informations to check how the test is going and in what stage the test is on
                        print(f"Iteration -> {indice} DB -> {name} Dim -> {dim} N -> {n_sample} Cancelled , method -> {method}, recall -> -, nlist -> {nlist} nprobe -> {nprobe}") 

    ## Write DataFrame 
    df_gpu.to_csv('replication.csv', index=False)


main()
