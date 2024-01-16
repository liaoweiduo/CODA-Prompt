#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090

module load cuda/12.1
module load anaconda/2022.10

source activate coda

date
echo "START unpack data"
tar -xf ../datasets_cobj.tar -C /dev/shm
ln -s /dev/shm/datasets .
rm data
mv datasets data
echo "END unpack data"

#mkdir ~/.cache
#mkdir ~/.cache/torch
#mkdir ~/.cache/torch/hub
#rm ~/.cache/torch/hub/checkpoints
#ln -s ../checkpoints ~/.cache/torch/hub/
date

sh experiments/cobj.sh

