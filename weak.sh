#!/bin/bash

#SBATCH --job-name=weakConvergence


#SBATCH -N 4 # number of nodes
#SBATCH -n 16 # number of cores

#SBATCH --mem 1024 # memory to be used per node
#SBATCH -t 16:00:00 # time (D-HH:MM)
#SBATCH --mail-type=START,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=xzhang@pppl.gov # send-to address

NPART=1024
L=8
TRIALS=32

python convTests.py -w EM $NPART $L 0.1 $TRIALS 0
for cor in 0 1
do
    python convTests.py -w MEM $NPART $L 0.1 $TRIALS $cor
done

