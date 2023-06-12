## Importing required libs

import numpy as np
import time
import sys
import faiss
from multiprocessing.pool import ThreadPool
import pandas as pd
import gc
import math





# python3 temp.py -ngpu 3 -nprobe 5,10,25,50,75,100 -ivf ivfpq -N 1000000


################################################################################################################
#                                                                                                              #
#                                           VARIABLE DEFINITIONS                                               #     
#                                                                                                              #         
#                   In this part, all the auxiliar variables will be defined                                   #
#                    (the function of each one is explained alongside the variable)                            #     
#                                                                                                              #
#                                                                                                              # 
################################################################################################################

args = sys.argv[1:]

ngpu = 3#number of gpus to be used
nprobes_in = [25]
index_key = 'ivfpq'
N = int(100e6)

while args:
    a = args.pop(0)
    if a == '-ngpu':      ngpu = int(args.pop(0))
    elif a == '-nprobe':  nprobes_in = [int(x) for x in args.pop(0).split(',')]
    elif a == '-ivf':     index_key = args.pop(0) # 'ivfflat','ivfpq'
    elif  a == '-N':      N = int(args.pop(0)) 
    else:
        print("argument %s unknown" % a, file=sys.stderr)
        sys.exit(1)

K = 20 # Number of neighbors to search

tempmem = -1 #1536*1024*1024 #-1 # option to reduce the temporary memory allocation 

altadd = False # avoids GPU memory overflows during add, IF true add wil not be done on GPU.

replicas = ngpu # Number of shardings

add_batch_size = 32768 # Batch add size (this can be increased)
search_batch_size = int(16384*3)
NORM = False

## Methods
metodos = {'deterministico': ['brute'],
           'aproximado':[index_key]
}

# Define the datasets n sample
dbs = [N]
index = 0
d = 12


assert faiss.get_num_gpus() >= ngpu

print(f"preparing resources for {ngpu} GPUs")


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
#                                           kNN SEARCHING Class                                                #            
#                           The Classes below are related with the kNN methods                                 #
#                           that are being tested in this benchmark. It is important to                        #
#                           say that there are 2 brute forces (sharding and replication)                       #
#                           this is done becuase for < 50M BF replication is faster                            #
#                                                                                                              #
################################################################################################################



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

        t0 = time.time()
        ## Search
        _, I = index.search(self.data, k)
        
        tf = time.time() - t0
        return sanitize(I),tf

################################################################################################################
#                                                                                                              #
#                                           SHARDING CLASS                                                     #     
#                                                                                                              #         
#                           This is the class that stores the main sharding attributes                         #     
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


################################################################################################################
#                                                                                                              #
#                                           SHARDING BF                                                        #     
#                                                                                                              #         
#                           This is the class that runs the sharding BF                                        #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################
        
class ShardingBrute(ShardedMultiGPUIndex):

    def __init__(self, data, name, gpu_resources):
        super().__init__(data, name, gpu_resources)

    
    ## This is the sharding method
    def search(self,K):

        ## Initiate timer
        t0 = time.time()

        ## N_sample

        nq_gt = self.N
        if self.N >= int(50e6):
            nq_gt = int(1e4)

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
        bs = search_batch_size


        ## Make sure that this is the database to search
        xqs = self.data[:nq_gt]

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
#                                           SHARDING ANN                                                       #     
#                                                                                                              #         
#                           This is the class that runs the sharding aNN                                       #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

class ShardingANN(ShardedMultiGPUIndex):

    def __init__(self, data, name, gpu_resources,nlist,nprobe,preproc):
        super().__init__(data, name, gpu_resources)

        self.nlist = nlist
        self.nprobe = nprobe
        self.preproc = preproc

################################################################################################################
#                                                                                                              #
#                                           PREPARING QUANTIZER                                                #     
#                                                                                                              #         
#                   This area is reserved for functions that control the QUANTIZER                             #
#                    and the batch iteration of datasets.                                                      #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

    def train_coarse_quantizer(self,x, k):

        
        ## Gets the dimension
        d = self.preproc.d_out

        ## Prepare the quantizer
        clus = faiss.Clustering(d, k)

        ## Set the maximum points per centroid
        clus.max_points_per_centroid = int(1e7)

        ## Instantiate the timer
        t0 = time.time()


        x = self.preproc.apply_py(sanitize(x))

        ## Set the GPU resources to train the quantizer
        vres, vdev = self.make_vres_vdev()

        ## Multi-GPU index
        index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, faiss.IndexFlatL2(d))

        ## Train the quantizer (k-means)
        clus.train(x, index)
        centroids = faiss.vector_float_to_array(clus.centroids)
        tf = time.time() - t0
        ## Return the centroids

        self.train_time = tf

        return centroids.reshape(k, d)

    def prepare_coarse_quantizer(self):

        nt = max(int(1e6), 256 * self.nlist)

        ## Training the quantizer
        centroids = self.train_coarse_quantizer(self.data[:nt], self.nlist)
        


        ## Add to the Index the centroids (PQ Technique)
        coarse_quantizer = faiss.IndexFlatL2(self.preproc.d_out)
        coarse_quantizer.add(centroids)

        return coarse_quantizer


################################################################################################################
#                                                                                                              #
#                                           TRAINING VECTORS                                                   #     
#                                                                                                              #         
#                                     In the function below the index is created                               #
#                                     and it is also trained.                                                  #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

    def prepare_trained_index(self):

        ## Dimension
        d = self.preproc.d_out

        ## M
        m = d


        ## Get the coarse quantizer
        coarse_quantizer = self.prepare_coarse_quantizer()

        idx_model = None
        if self.name == 'ivfpq':
            ## HERE IT IS POSSIBLE TO CHANGE THE INDEX THAT WILL BE USED
            idx_model = faiss.IndexIVFPQ(coarse_quantizer,d , self.nlist, m, 8)

        else:
            idx_model = faiss.IndexIVFFlat(coarse_quantizer,d,self.nlist) 


        coarse_quantizer.this.disown()

        # finish training on CPU
        ## Here the training wil always be with 1e6 points, because this apporach is only used for datasets that have 60M plus vectors
        x = self.preproc.apply_py(sanitize(self.data[:int(1e6)]))

        t0 = time.time()
        idx_model.train(x)

        self.train_time += (time.time() - t0)

        return idx_model



################################################################################################################
#                                                                                                              #
#                                           GET THE INDEX                                                      #     
#                                                                                                              #         
#                                     In the function below the index is created                               #
#                                     and it is also sharded.                                                  #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################


    def compute_populated_index(self):

        """
        Add elements to a sharded index. 
        Return the index and if available
        a sharded gpu_index that contains the same data. 
        """

        ## Get the trained index (IVFFLAT, IVFPQ, IVFSQ)
        indexall = self.prepare_trained_index()

        
        ## Set the setting of Multi-GPU Execution
        co = faiss.GpuMultipleClonerOptions()
        co.usePrecomputed = False
        co.indicesOptions = faiss.INDICES_CPU
        co.verbose = False #True # Understand what's happening
        co.shard = True # Sharding


        vres, vdev = self.make_vres_vdev() # Get the resources for 4 GPUS

        ## Make the Multi-GPU Index with sharding
        gpu_index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, indexall, co)

        ## Now it is time to add the vector to the index, to be possible to search the kNN



        ## Number of vectors to add
        nb = self.N

        t0 = time.time()
        ## Adding in batches
        for i0, xs in self.dataset_iterator(self.data, self.preproc, add_batch_size):
            
            ## Get the end of the batch
            i1 = i0 + xs.shape[0]

            ## Add the index with ids, to speed up the process
            gpu_index.add_with_ids(xs, np.arange(i0, i1))
            
        self.add_time = time.time() - t0


        t0 = time.time()
        ## This is made because in other function it is possible to overwrite the GPU index, so it is necessary to save the indices
        if hasattr(gpu_index, 'at'):
            # it is a sharded index
            for i in range(ngpu):
                index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
                index_src.copy_subset_to(indexall, 0, 0, nb)

        self.aux_time = time.time() - t0

        return gpu_index, indexall


    def compute_populated_index_2(self):

        indexall = self.prepare_trained_index()


        # set up a 3-stage pipeline that does:
        # - stage 1: load + preproc
        # - stage 2: assign on GPU
        # - stage 3: add to index

        stage1 = self.dataset_iterator(self.data, self.preproc, add_batch_size)

        vres, vdev = self.make_vres_vdev()
        coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, indexall.quantizer)

        def quantize(args):
            (i0, xs) = args
            _, assign = coarse_quantizer_gpu.search(xs, 1)
            return i0, xs, assign.ravel()

        stage2 = rate_limited_imap(quantize, stage1)



        for i0, xs, assign in stage2:
            i1 = i0 + xs.shape[0]
            if indexall.__class__ == faiss.IndexIVFPQ:
                indexall.add_core_o(i1 - i0, faiss.swig_ptr(xs),
                                    None, None, faiss.swig_ptr(assign))
                
            elif indexall.__class__ == faiss.IndexIVFFlat:
                indexall.add_core(i1 - i0, faiss.swig_ptr(xs), None,
                                faiss.swig_ptr(assign))

        return None, indexall


    ## Gets the index that will be used to search the kNN
    ## Then make the shards that will be used in the search
    def get_populated_index(self):

        ## Decidir se os indices ficarão adicionados em GPU ou em CPU (caso não caibam em GPU de nenhuma maneira --> 1 BILHÃO)
        if not altadd:
            gpu_index, indexall = self.compute_populated_index()
        
        
        else:
            gpu_index, indexall = self.compute_populated_index_2()
        

        ## Multi-GPU configuration
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = False
        co.useFloat16CoarseQuantizer = False
        co.usePrecomputed = False
        co.indicesOptions = 0
        co.verbose = False
        co.shard = True 

        t0 = time.time()

        ## Move index to GPU if there is no GPU Index


        # We override the GPU index
        del gpu_index 


        ## Make de sharding
        index = faiss.IndexReplicas()


        for i in range(replicas):

            ## Get gpus
            gpu0 = ngpu * i // replicas
            gpu1 = ngpu * (i + 1) // replicas

            ## Get resources for the gpus
            vres, vdev = self.make_vres_vdev(gpu0, gpu1)


            ## Index multiGPU
            index1 = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
            
            ##Ading the index to the sharding
            index1.this.disown()
            index.addIndex(index1)

        self.aux_time += (time.time() - t0)
        del indexall

        self.index = index


################################################################################################################
#                                                                                                              #
#                                           kNN SEARCHING                                                      #     
#                                                                                                              #         
#                   The function below runs the search of the kNN in batches, for many nprobes                 #
#                    values, iterating the dataset by the batch_size (sl). This function can be modified       #     
#                     to return the indices and the distances of the kNN to build the kNNG.                    #
#                                                                                                              # 
################################################################################################################
 

    def search(self,K):

        ## Set the index parameters easier
        ps = faiss.GpuParameterSpace()
        ps.initialize(self.index)


        ## Set the number of queries (in this case the same size of N)
        nq = self.N


        ## Set nprobe to the index
        ps.set_index_parameter(self.index, 'nprobe', self.nprobe)

        ## Initiate the timer
        t0 = time.time()

        ## Instanciate the indices and distances arrays
        I = np.empty((nq, K), dtype='int32')
        D = np.empty((nq, K), dtype='float32')

        ## Auxiliar variable
        inter_res = ''

        ## Dataset iterator return the indices and the dataset, so i0 = indices and xs = batched array
        for i0, xs in self.dataset_iterator(self.data, self.preproc, search_batch_size):

            ## Show in which stage is the execution (time)
            print(f"\r{i0}/{nq} ({time.time()-t0:.3f} s{inter_res}) - {self.name}", end=' ')

            
            sys.stdout.flush()

            ## Set the final position of the array
            i1 = i0 + xs.shape[0]

            ## Perform the search in batch
            Di, Ii = self.index.search(xs, K)

            ## Assing the distances/indices calculated to the primary array
            I[i0:i1] = Ii
            D[i0:i1] = Di

        ## Total time
        t1 = time.time()

        ## Return
        return I,(t1-t0)

################################################################################################################
#                                                                                                              #
#                                           AUXILIAR FUNCTIONS                                                 #     
#                                                                                                              #         
#                           The functions below are related with auxiliar methods                              #
#                                                                                                              #
#                                                                                                              # 
################################################################################################################

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


def eval_intersection_measure(gt_I, I):
   #measure intersection measure (used for knngraph)

    n,k = gt_I.shape
    recall_value = (gt_I[:,:] == I[:,:]).sum() / float(n*k)

    return recall_value

def recall(gt_I,I,k):

    
    #Somatório dos k primeiros vizinhos positivos dividido por n*k
    n = I.shape[0]

    if n >= int(50e6):
        return eval_intersection_measure(gt_I[:,:k], I[:int(1e4),:k])


    recall_value = (gt_I[:,:k] == I[:,:k]).sum() / (float(n*k))
    
    return recall_value



def create_object(name,info):

    index = None




    """

        This function have some variables to instanciate an object:
            - Name == 'ivfpq' | 'ivfflat' -> Instanciante an aNNSharding
            - Name == 'brute' -> If N_sample <= 30M, instanciante the standart 
            one BF, else instanciate the sharding one
    """

    ## Create brute force index

    if name == 'brute':

        if info['N_sample'] <= int(30e6):
            index = MultiBrute(info['data'],name)
        else:
            index = ShardingBrute(info['data'],name,info['gpu_res'])

        return index

    preproc = get_preprocessor(info['data'])

    index = ShardingANN(info['data'],name,info['gpu_res'],info['nList'],info['nprobe'],preproc)
    
    return index



################################################################################################################
#                                                                                                              #
#                                           MAIN EXECUTION                                                     #     
#                                                                                                              #         
#                   Here, the main execution of the test is done, this is the main()                           #
#                                                                                                              #           
#                                                                                                              #
#                                                                                                              # 
################################################################################################################



def main():
    

    df_gpu = None

    file_name = 'sharding.csv'
    if N > int(0.6e6):
        file_name = 'large_sharding.csv'
    ## Initializating the dataframe
    try:
        
        df_gpu = pd.read_csv(file_name)
    except:
        print("DF_GPU ainda nao existe, logo vai ser criado")
        df_gpu = pd.DataFrame()

    ## Control variable
    indice = df_gpu.shape[0]
    begin = indice



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
            gpu_resources = tempMem(tempmem)
            for method in metodos[c]: #brute, ivfflat,ivfpq,ivfsq
                ## Warm-UP GPU
                if indice == begin:
                    print("Warming up the GPU...")
                    
                    _,_ = MultiBrute(data[:int(1e6)],method).search(K)
                    print("GPU Ready to go...")
                    time.sleep(5)

                """
                Here, all the possible configurations that the approximate methods can be executed will be tested, and their results will be inserted
                on the list above. It will be possible to order the results by time or recall
                """

                ## Setting nlists and nprobes values for the approximate method
                nlist = int( 2 ** ( 2 * round(math.log(n,10)) ) )

                nprobes = nprobes_in

                ## Setting random values for brute method
                if method == 'brute':
                    nlist = 0
                    nprobes = [0]

                ## Info declaration


                ## Primary iteration
                for nprobe in nprobes:
                    
                    info = {}

                    info['gpu_res'] = gpu_resources

                    ## Writing control variable
                    indice += 1

                    ## Set the information about the methdo, if the brute method is performed nlist and nprobe will not be used


                    n_sample,dim = (n,d)

                    info['Name'] = name
                    info['Method'] = method
                    info['Dim'] = dim
                    info['N_sample'] = n_sample
                    
                    info['gpu_res'] = gpu_resources

                    info['nList'] = nlist
                    info['nprobe'] = nprobe
                    info['DB Size (GB)'] = size
                    info['data'] = data



                    try:

                        ## Create the object by the name 
                        index = create_object(method,info)

                        if method != 'brute':
                            index.get_populated_index()
                            info['Train time'] = index.train_time
                            info['Add time'] = index.add_time
                            info['Move time'] = index.aux_time
                            info['Total'] = info['Train time'] + info['Add time'] + info['Move time']
                        
                        ## Perform the search, saving the indices and the kNN time
                        indices,time_knn = index.search(K)
                
                        rec_value = '-'

                        ## Save the exact result
                        if method == 'brute':
                            brute_indices = indices.copy()
                            s = f'Recall@{rec_k}'
                            info['Total'] = time_knn

                            info['Time kNN'] = time_knn
                            info[s] = rec_value

                        ## Calculate the results and ADD them to the list
                        if method != 'brute':
                            rec_value = recall(brute_indices,indices,rec_k)
                            s = f'Recall@{rec_k}'
                            info['Time kNN'] = time_knn
                            info[s] = rec_value
                            info['Total'] += time_knn

                    

                        write_df(df_gpu,indice,info)

                        #Show the informations to check how the test is going and in what stage the test is on
                        print(f"Iteration -> {indice} DB -> {name} Dim -> {dim} N -> {n_sample} Finished in {time_knn:.5} secs, method -> {method}, recall -> {rec_value}, nlist -> {nlist} nprobe -> {nprobe} TOTAL ->{info['Total']:.3}")            
                        
                        del index,indices
                    except Exception as e:
                        
                        s = f'Recall@{rec_k}'
                        info = {'Time kNN':'-',s:'-','nList':nlist,'nprobe':nprobe}
                        info['Name'] = name
                        info['Method'] = method
                        info['Dim'] = dim
                        info['N_sample'] = n_sample
                        info['DB Size (GB)'] = size
                        info['Erro'] = str(e)

                        write_df(df_gpu,indice,info)

                        #Show the informations to check how the test is going and in what stage the test is on
                        print(f"Iteration -> {indice} DB -> {name} Dim -> {dim} N -> {n_sample} Cancelled , method -> {method}, recall -> -, nlist -> {nlist} nprobe -> {nprobe}") 
                        time.sleep(2)
                del gpu_resources
        del brute_indices
    ## Write DataFrame 
    df_gpu.to_csv(file_name, index=False)


main()
gc.collect()


