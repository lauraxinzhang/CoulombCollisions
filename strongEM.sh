#!/bin/bash

#SBATCH --job-name=strongEM



#SBATCH -n 64 # number of cores

#SBATCH --mem 1024 # memory to be used per node
#SBATCH -t 36:00:00 # time (D-HH:MM)
#SBATCH --mail-type=START,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=xzhang@pppl.gov # send-to address

source activate coulomb
NPART=521
L=9
TRIALS=0
TTOT=0.01

python convTests.py -s EM $NPART $L $TTOT 0

