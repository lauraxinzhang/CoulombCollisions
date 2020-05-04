#!/bin/bash

#SBATCH --job-name=weak5

#SBATCH -n 64 # number of cores

#SBATCH --mem 1024 # memory to be used per node
#SBATCH -t 48:00:00 # time (D-HH:MM)
#SBATCH --mail-type=START,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=xzhang@pppl.gov # send-to address

NPART=8192
L=10
TRIALS=32
TTOT=0.01

#python convTests.py -w EM $NPART $L $TTOT $TRIALS 0
for cor in 5
do
    python convTests.py -w MEM $NPART $L $TTOT $TRIALS $cor
done

