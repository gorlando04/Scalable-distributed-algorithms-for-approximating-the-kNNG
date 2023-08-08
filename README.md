# Scalable Distributed Algorithms for approximating the kNNG

Finding a dataset's k Nearest Neighbors Graph (kNNG) is an essential estimation in several algorithms, such as clustering and anomaly detection. However, this operation is costly since when calculating all k neighbors of all points in the set, there is a quadratic complexity concerning the size of the data set. Thus, there is a physical limitation in the execution of these algorithms since if large amounts of data are used, the algorithm tends to become slower, given the complexity of memory and the complexity of time. Due to this limitation, the existence of kNNG algorithms that build the approximate graph but, at the same time, maintain quality and are scalable is fundamental. Therefore, an idea that allows the fulfillment of these requirements is the distribution and parallelization of the algorithms that find the approximate kNNG. Thus, checking the scalability of approximate kNNG algorithms is essential to determine which algorithms have the best results in terms of computational cost and quality of the result and can be parallel and distributed. The present paper encompasses the theoretical comparison, based on the computational complexity of the algorithms presented, such as Inverted File algorithms and NNDescent, and the practical comparison, based on the algorithm's precision and accuracy, regarding their scalability. With the study presented in this paper, checking for the algorithm's scalability and finding the most performative method that constructs the approximate kNNG in terms of time and quality of the resulting kNNG will be possible. Therefore this study can speed up many machine learning techniques that build the kNNG, allowing these techniques to handle voluminous datasets.

In this repository the implementation to perform this comparison can be found, also, the instructions to run and software specifications can be found.

## Software Especifications

The comparsion was performed in a GPU server with the following specifications

<aside>
💡 RAM: 128 GB
	
💡 Processor: 2 Intel(R) Xeon(R) Silver 4208 CPU

💡 GPU: 3 NVIDIA GeForce RTX 2080 Super

💡 Ubuntu 22.04

💡 CUDA 12.0

💡 Docker Version: 20.10.21

</aside>

## Docker images 

In this experiment, two containers were build, one for FAISS and other for NNDescent.

### FAISS

To build the container for FAISS , the following image was used:

```
docker run --gpus '"device=0,1,2"' --name RAPIDS_CONTAINER --rm -it -v /home/gabriel:/rapids -w /rapids --shm-size=1g --ulimit memlock=-1 [nvcr.io/nvidia/rapidsai/rapidsai-core:22.10-cuda11.5-base-ubuntu18.04-py3.8](http://nvcr.io/nvidia/rapidsai/rapidsai-core:22.10-cuda11.5-base-ubuntu18.04-py3.8)
```

When acessing the container, the following steps were done:

```bash
#Installing Nano
apt update
apt install nano
apt install -y tmux
clear 

# Deactivating conda environment
conda deactivate

# Creating new environment
conda create -y -n faiss python=3.9
conda activate faiss

# Installing
conda install -y pytorch==1.8 -c pytorch
conda install -y faiss-gpu=1.6 -c pytorch
conda install -y pandas
conda install -y scikit-learn
conda install -y hdbscan -c conda-forge
pip install matplotlib
clear
```

After the beggining of every execution, it was tested if the container was with GPUs working with the **************nvidia-smi************** command. Also it is important to say that the FAISS-GPU used was 1.6

### NNDescent

To build the container for NNDescent , the following image was used:

```
docker run --gpus '"device=0,1,2"' --name GPU_NNDESCENT --rm -it -v /home/gabriel:/nndescent -w /nndescent --shm-size=1g --ulimit memlock=-1 nvidia/cuda:12.0.1-devel-ubuntu22.04
```

When acessing the container, the following steps were done:

```bash
#Installing Nano
apt update
apt install nano
apt install -y tmux
clear 

apt install -y cmake

apt install -y wget

apt install -y python

apt install -y python3-pip

pip install scikit-learn

pip install numpy

pip install pandas


```

After the beggining of every execution, it was tested if the container was with GPUs working with the **************nvidia-smi************** command. 

## Experiments

The experiments are presented in 
[benchmarking-FAISS](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/tree/main/benchmarking-FAISS)  and 
[benchmarking-NNDescent](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/tree/main/benchmarking-NNDescent) , 
and need to be done in different containers due to different software requirements.




## Conclusion


The paper fulfilled its goals, introducing and studying multiple kNNG algorithms. Scalability was tested in two scenarios: replication, which faced memory limitation issues and
sharding, which was able to handle larger datasets.
In the first scenario, IVFFlat, NNDescent, and IVFSQ had significant GPU memory usage, which made them unable to construct the kNNG for datasets over 20M samples. However, NNDescent and IVFFlat had high-quality approximations, having more
than 0.95 of Recall@10. IVFPQ was rapid but lower quality than IVFFlat and NNDescent. NNDescent I/O performed well, constructing kNNG for all datasets in adequate time
when compared to the exact-construction, and had a high recall, having more than 0.86 for
most datasets. In the second scenario, IVFPQ constructed kNNG for large-scale datasets
quickly with multi-GPU and Sharding benefits, 74x faster than exact-construction . However, IVFPQ had lower quality, while NNDescent I/O provided good speed, compared to
the exact-construction, 42x faster than exact-construction, and high-quality approximations with Recall@10 over 0.92. Therefore, it indicates a trade-off between speed and
quality when using IVFPQ and NNDescent I/O.
In the future, we plan to use the results of this research to enhance several machine
learning techniques that relies on the construction of the kNNG. Our findings will guide
our choices of algorithm and parameters based in the scenario of application and dataset
characteristics.
