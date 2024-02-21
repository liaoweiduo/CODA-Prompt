#!/bin/bash
#SBATCH --error=output/job.%j.err
#SBATCH --output output/job.%j.out
#SBATCH -p v100
#SBATCH --qos=v100
#SBATCH -J lwd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2

ulimit -n 10000

cd datasets/CODA-Prompt

echo "START map data"
ln -s '../datasets' .
rm data
mv datasets data
echo "END map data"

date
singularity exec --nv ~/sif/coda-2.0_sklearn.sif bash experiments/cgqa.sh
date