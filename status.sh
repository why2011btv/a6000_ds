#!/bin/bash
#SBATCH --job-name=LEC_OnePass
#SBATCH --nodelist=nlpgpu05
#SBATCH --output=./nvidia.out
nvidia-smi
ps -ef |grep why