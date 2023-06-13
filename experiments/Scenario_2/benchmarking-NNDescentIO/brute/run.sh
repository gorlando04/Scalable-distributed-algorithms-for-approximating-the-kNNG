#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1

	N=100000000
	step=50000000
        for i in {1..7}
        do 

        python3 brute.py $N

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