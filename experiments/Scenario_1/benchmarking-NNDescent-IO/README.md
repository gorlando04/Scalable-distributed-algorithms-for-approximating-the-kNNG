# NNDescent

This repository presents a modification on the Source code for CIKM 2021 paper [Fast k-NN Graph Construction by GPU based NN-Descent](https://dl.acm.org/doi/10.1145/3459637.3482344). Implementing NNDescent on multi-GPU, but using the Merge algorithm presented on [Fast k-NN Graph Construction by GPU based NN-Descent](https://dl.acm.org/doi/10.1145/3459637.3482344). This code was done to be ran on 3 GPUs,

## Observations

Firstly, it is important to say that in order to compile correctly the source code it is important to follow this instructions:

1. Check GPU compute capability in [NVIDIA](https://developer.nvidia.com/cuda-gpus). After that, it is important to change the value for the correct compute capability in [CMakeLists.txt](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/benchmarking-NNDescent/CMakeLists.txt). In the CMake file, the following value must be changed to the correct compute capability:

```
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FILE_OFFSET_BITS=64 -O3 -std=c++14 -arch=sm_(COMPUTE_CAPABILITY) -rdc=true -Xcompiler -fopenmp -pthread")
```

2. After that the following commands must be done:

```
cd cmake
cmake ..
make
```

3. Finally, the executable file will be avaiable.


## Parameters

It is important to say that in [nndescent.cuh](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/benchmarking-NNDescent/gpuknn/nndescent.cuh) the primary parameters of the algorithm are set. This is done in the first lines.

```cpp
const int VEC_DIM = 12; // Vectors dimension
const int NEIGHB_NUM_PER_LIST = 32; //Value of K in kNN
const int SAMPLE_NUM = 16;  // assert(SAMPLE_NUM * 2 <= NEIGHB_NUM_PER_LIST);
const int NND_ITERATION = 6; // Iterations of the algorithm
const int MERGE_SAMPLE_NUM = 12;
const int MERGE_ITERATION = 11;
```

Also, in [main.cu](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/benchmarking-NNDescent/main.cu) we have some parameters that must be change ir order to run the algorithm correctly. The first one is on the beggining of the source code

```cpp
#define N_SAMPLE 1000000

```

Which indicates the size of the dataset that will be used. Additionally, the following parameters must be changed:

```cpp
    string base_path = "/nndescent/GPU_KNNG/data/artificial/SK-1M_data.txt";

```

Indicating the dataset that will be used on the experiment.

## Data

Data can be created by two ways: by running [push.sh](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/benchmarking-NNDescent/data/push.sh) or by running [create.py](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/benchmarking-NNDescent/data/artificial/create.py) using the following command:

```
python3 create.py N_SAMPLE
```
## Results

The result were checked after the end of the algorithms run. To check the Recall@10 of the kNNG it was necessary to run [brute.py](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/benchmarking-NNDescent/brute/brute.py), however this script must be run on FAISS-container as it uses FAISS-BF method. To run this script the following command needs to be done:

```
python3 brute.py N_SAMPLE
```


## Reference

The repository that was used as an inspirations to this research is [GPU_KNNG](https://github.com/RayWang96/GPU_KNNG)





