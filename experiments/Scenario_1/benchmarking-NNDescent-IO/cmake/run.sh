#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1

	N=0
  	i=0
	step=10000000
        for shards in '3' '3' '3' '6' '6' '9' 
        do 
        if [ $N = '0' ]
	then
		N=1000000
	fi
	echo $N

        ## Cria o dataset em .txt
        python3 /nndescent/GPU_KNNG/data/artificial/create.py $N
        sleep 1
        
        ## Transforma o dataset para um arquivo binário
        ./gknng 
        sleep 5

        mkdir /nndescent/GPU_KNNG/results/Test$i

        ## Roda o NNDescent
        ./gknng false $shards $N > /nndescent/GPU_KNNG/results/Test$i/Result.txt

        python3 eval.py $N > /nndescent/GPU_KNNG/results/Test$i/Recall.txt
        sleep 2


        # Remover os resultados

        rm /nndescent/GPU_KNNG/data/artificial/SK_data.txt
        rm /nndescent/GPU_KNNG/data/vectors.*
        rm /nndescent/GPU_KNNG/results/NNDescent-KNNG.*

        if [ $N = '1000000' ]
        then
                N=0
        fi


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
