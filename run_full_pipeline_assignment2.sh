#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=Multi_Assign2
#SBATCH --mail-type=END
#SBATCH --mail-user=karen.lopez@ucalgary.ca
#SBATCH --output=Output_full_pipeline_assigment2_%j.out

echo Starting slurm script 

date 
id 

echo Start initialization 

source /home/karen.lopez/software/miniconda3/etc/profile.d/conda.sh
conda activate ENEN645_gpu

which python
conda env list

# Sanity check GPU visibility
nvidia-smi
python -c "import torch; print(torch.__version__); print('CUDA avail:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

echo Finished initializing

# Execute the Python code
python full_pipeline_assignment2.py

echo Ending slurm script for full_pipeline_assignment2.py

date