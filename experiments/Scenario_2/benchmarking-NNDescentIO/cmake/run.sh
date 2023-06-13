#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1

	N=100000000
  i=0
	step=50000000
        for shards in '15' '24' '30' '39' '45' '54' '60'
        do 

        ## Cria o dataset em .txt
        python3 /nndescent/GPU_KNNG/data/artificial/create.py $N
        sleep 1
        
        ## Transforma o dataset para um arquivo binário
        ./gknng 
        sleep 5

        mkdir /nndescent/GPU_KNNG/results/Test$i

        ## Roda o NNDescent
        ./gknng false $shards > /nndescent/GPU_KNNG/results/Test$i/Result.txt

        python3 eval.py $N > /nndescent/GPU_KNNG/results/Test$i/Recall-Begin.txt
        sleep 2

        python3 eval.py $N -F > /nndescent/GPU_KNNG/results/Test$i/Recall-Final.txt
        sleep 2; 

        # Remover os resultados

        rm /nndescent/GPU_KNNG/data/artificial/SK_data.txt
        rm /nndescent/GPU_KNNG/data/vectors.*
        rm /nndescent/GPU_KNNG/results/NNDescent-KNNG.*



         N=`expr $N + $step`
         i=`expr $i + 1`
         done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit