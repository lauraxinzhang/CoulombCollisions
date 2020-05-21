#!/bin/bash

source activate coulomb

for cor in 0 1 2 5
do
    export ORDER=$cor
    sbatch strong.sh
    #echo $ORDER
done

#sbatch strongEM.sh
