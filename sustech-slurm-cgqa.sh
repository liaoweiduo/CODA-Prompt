#!/bin/bash
#SBATCH --error=~/output/job.%j.err
#SBATCH --output ~/output/job.%j.out
#SBATCH -p v100
#SBATCH --qos=v100
#SBATCH -J lwd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

ulimit -n 10000

date
singularity exec --bind ~/datasets/datasets:data --nv ~/sif/coda.sif bash experiments/cgqa-ft.sh
date
