#!/bin/bash

#SBATCH --job-name=weak

#SBATCH -n 64 # number of cores

#SBATCH --mem 4096 # memory to be used per node
#SBATCH -t 48:00:00 # time (D-HH:MM)
#SBATCH --mail-type=START,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=xzhang@pppl.gov # send-to address

source activate coulomb

NPART=8192
L=7
TRIALS=128
TTOT=0.1
ORDER=0

#ORDER=5
#python convTests.py -w EM $NPART $L $TTOT $TRIALS 0
python convTests.py -w MEM $NPART $L $TTOT $TRIALS $ORDER

