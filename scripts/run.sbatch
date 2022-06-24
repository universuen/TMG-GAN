#!/bin/bash

#!/bin/bash

#SBATCH --account=cuixiaohui
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=3:30:00

conda activate tmg_gan

cd $SLURM_SUBMIT_DIR

python train_wgan.py
