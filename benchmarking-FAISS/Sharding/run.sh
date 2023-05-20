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