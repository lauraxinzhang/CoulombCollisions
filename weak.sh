#!/bin/bash

#SBATCH --job-name=weakConvergence


#SBATCH -N 4 # number of nodes
#SBATCH -n 16 # number of cores

#SBATCH --mem 64 # memory to be used per node
#SBATCH -t 48:00:00 # time (D-HH:MM)
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=xzhang@pppl.gov # send-to address

python convTests.py -w EM 1024 8 0.1 16 0
for cor in [0, 1, 2, 5]
do
    python convTests.py -w MEM 1024 8 0.1 16 cor
done

