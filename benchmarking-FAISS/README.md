# Scalable Distributed Algorithms for approximating the kNNG

Finding the K Nearest Neighbors Graph (kNNG) of a dataset is an essential estimation used in several algorithms, such as clustering and anomaly detection. However, this operation is costly since when calculating all k neighbors of all points in the set, there is a quadratic complexity concerning the size of the data set. Thus, there is a physical limitation in the execution of these algorithms since if large amounts of data are used, the algorithm tends to become slower, given the complexity of memory and the complexity of time. Due to this limitation, the existence of kNNG algorithms that build the approximate graph but, at the same time, maintain quality and are scalable is fundamental. Therefore, an idea that allows the fulfillment of these requirements is the distribution and parallelization of the algorithms that find the approximate kNNG. Thus, to find out which algorithms that build the approximate kNNG have the best results in terms of computational cost and quality of the result and can be parallel and distributed, it is necessary to carry out a comparative study between these methods. This study encompasses the theoretical comparison, based on the computational complexity of the algorithms, and the practical comparison, based on the algorithm's precision, accuracy, and the study of the advantages and disadvantages of each algorithm that builds the approximate kNNG.

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

To build the container, the following image was used:

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

## Experiments

The experiments are splited into two parts: Sharding and Replication

### Replication

Replication’s execution was more simple, and could be done in one command:

```bash
## Executing Replication benchmarking
python3 replication.py
```

The output of this test was store in a .csv file that in the future was used to create the graphics.

### Sharding

Sharding’s execution were incremental, so the same script was executed 12 times, each time with a dataset and algorithm (IVFFlat and IVFPQ) different, and the results were store in a .csv that in the future was analyzed. The following code was responsible for Sharding’s execution. This code is the [run.sh](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/tree/main/benchmarking/Sharding/run.sh) file. It is important to note that this file must be executable, and to this, just chmod +x run.sh, and then ./run.sh

```bash
#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){
	sleep 1
	N=0
	step=10000000
        for i in {1..6}
        do 
         for j in 'ivfflat' 'ivfpq'
         	do
         	python3 sharding.py -ngpu 3 -nprobe 5,10,25,50,75,100 -ivf $j -N $N
         	sleep 2; 
         	done
         N=`expr $N + $step`
         done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
```

## Data

The data used was created during the execution and can be replicated as the skelarn’s methods have a seed.

## Results

The .csv files were used in a jupyte-notebook to create the graphics that were used on the paper, the way the figures were created and all the steps are store in [results](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/tree/main/results).

## Conclusion

This paper presents a comparison between approximate methods that construct the kNNG in multi-GPU systems to benefit from the parallelization and the distribution to speed them up. First of all, it was presented how each algorithm used in this survey worked, defining the idea of Inverted File and Product Quantization and how both allow the algorithms to approximate the \textit{k}NNG and control the memory usage. Also, both GPU execution techniques were defined, Replication and Sharding, and the benefits and drawbacks of each method, exploring the idea that Sharding can handle larger datasets due to splitting them into the GPUs. At the same time, Replication only works if the dataset can be entirely held in GPU memory, as it creates replicas of the dataset of the GPUs. Finally, all the algorithms are compared, as it was proposed, using different techniques and datasets to understand the performance of each one and to check the quality of the approximate kNNG constructed for each algorithm. Finally, in the experiments, it could be observed that IVFPQ is an excellent algorithm to build the approximate kNNG with larger datasets. In contrast, the other two algorithms (IVFFlat and IVFSQ) are not appropriate for constructing the approximate \textit{k}NNG, suffering from memory constraints when the dataset is large. In conclusion, as the idea of this paper was to find the best approximate algorithm that constructs the \textit{k}NNG in a multi-GPU system, with all the studies and experiments, IVFPQ had the best performance in terms of memory, time, and approximate kNNG quality and also can handle larger datasets, such as datasets over 100 million samples.
