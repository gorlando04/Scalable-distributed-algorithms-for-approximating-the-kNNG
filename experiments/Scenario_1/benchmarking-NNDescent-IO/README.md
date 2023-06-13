# NNDescent-I/O

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

## Experiments

We created a bash script [run.sh](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/experiments/Scenario_2/benchmarking-NNDescentIO/cmake/run.sh) to run the experiments and avoiding disk overflow. So, we only need to create all the exact kNNG for all the datasets. This can be done running [run.sh](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/experiments/Scenario_2/benchmarking-NNDescentIO/brute/run.sh) in the FAISS container, because it uses different software specifications. So, after building the exact kNNG for all the datasets we can simply run [run.sh](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/blob/main/experiments/Scenario_2/benchmarking-NNDescentIO/cmake/run.sh).

The script pesented in [cmake](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/tree/main/experiments/Scenario_2/benchmarking-NNDescentIO/cmake) builds the datasets, transform the dataset for being avaible to be process and evaluate the results, and store the results in a folder called Test$i, where i goes from 0 to 6.

It is important that we have enough space on disk to build the kNNG, else the script is not going to work.




## Reference

The repository that was used as an inspirations to this research is [GPU_KNNG](https://github.com/RayWang96/GPU_KNNG)





