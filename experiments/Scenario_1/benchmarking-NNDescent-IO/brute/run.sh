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
	echo "$N"
        python3 brute.py -N $N
        sleep 2; 

         N=`expr $N + $step`
         done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit

