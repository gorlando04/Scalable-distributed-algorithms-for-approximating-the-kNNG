# Experiments

We have all the experiments described in this repository. As it has been said in the paper, we presents two scenarios where we compare the performance of many kNNG algorithms in multi-GPU systems. Althoguh theses scenarios are described in the paper, we can explain them.

##Scenario 1

The first scenario is where the dataset must fit into the memory of one GPU, and therefore it can be replicated over the GPUs and the queries can be parallelized over the GPUs speeding up the construction.

For this scenario, we have many algorithms, such as NNDescent, NNDescent/IO, IVFPQ, IVFFLat and IVFSQ, and also we teste brute force algorithm to check how much time it takes to build the exact kNNG for the datasets.


##Scenario 2

The second scenario is where the dataset dont fit into the memory of one GPU, and therefore it must be split over the GPUs to be processed, which increase the ammount of memory that can be hold simultaneously by the GPUs.

For this scenario we have 2 algorithms, NNDescent-I/O and IVFPQ, and also we teste brute force algorithm for some datasets, setting a threshold of time of 5 days, which lead us able to build the exact kNNG for datasets with 100M, 150M and 200M samples.

All the instructions are described in the scenario folders.
