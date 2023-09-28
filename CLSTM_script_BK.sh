#!/usr/bin/bash

module load cuda
LD_LIBRARY_PATH="/usr/local/cuda-11.0.3/lib64:/wynton/protected/home/ibrahim/alex_schubert/lib64"
export LD_LIBRARY_PATH

source ~/.bashrc #configures your shell to use conda activate
conda activate PCR_env

export CUDA_VISIBLE_DEVICES=$SGE_GPU

gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID

python 

# stats
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof