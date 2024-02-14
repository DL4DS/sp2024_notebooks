#!/bin/bash -l

# Set SCC project
#$ -P ds598


module load miniconda/4.9.2
conda activate ds598 # activate your conda environment

python gpu_training.py

# To submit the job to SCC, run the following command
# qsub -pe omp 4 -P ds598 -l gpus=1 -o output.txt -e error.txt run.sh