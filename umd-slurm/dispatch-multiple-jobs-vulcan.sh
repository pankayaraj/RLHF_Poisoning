#!/bin/bash
#SBATCH --job-name=pytorchjob
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=16gb
#SBATCH --qos=scavenger
#SBATCH --time=10:00:00
#SBATCH --output=/vulcanscratch/gihan/umd-slurm/logs/outFile-%A_%a.txt
#SBATCH --error=/vulcanscratch/gihan/umd-slurm/logs/errorFile-%A_%a.txt
#SBATCH --array=1-16



CONDA_ENV_NAME = "longtails"

source ~/.bashrc
export TORCH_HOME=/vulcanscratch/gihan/torch-hub/
conda activate $CONDA_ENV_NAME


sed -n "${SLURM_ARRAY_TASK_ID}p" < /vulcanscratch/gihan/umd-slurm/list-of-commands.sh
sed -n "${SLURM_ARRAY_TASK_ID}p" < /vulcanscratch/gihan/umd-slurm/list-of-commands.sh >&2
echo "------"
echo "------" >&2
eval $(sed -n "${SLURM_ARRAY_TASK_ID}p" < /vulcanscratch/gihan/umd-slurm/list-of-commands.sh)



echo "END of SLURM commands"
echo "END of SLURM commands" >&2