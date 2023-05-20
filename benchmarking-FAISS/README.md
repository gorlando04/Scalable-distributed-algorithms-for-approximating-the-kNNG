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

Sharding’s execution were incremental, so the same script was executed 12 times, each time with a dataset and algorithm (IVFFlat and IVFPQ) different, and the results were store in a .csv that in the future was analyzed. The following code was responsible for Sharding’s execution. This code is the [run.sh](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG/tree/main/benchmarking-FAISS/Sharding/run.sh) file. It is important to note that this file must be executable, and to this, just chmod +x run.sh, and then ./run.sh

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


