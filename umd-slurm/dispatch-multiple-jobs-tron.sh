#!/bin/bash
#SBATCH --job-name=pytorchjob
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --output=/cmlscratch/pan/umd-slurm/logs/outFile-%A_%a.txt
#SBATCH --error=/cmlscratch/pan/umd-slurm/logs/errorFile-%A_%a.txt
#SBATCH --array=1-8




source ~/.bashrc


sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh
sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh >&2
echo "------"
echo "------" >&2
eval $(sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh)



echo "END of SLURM commands"
echo "END of SLURM commands" >&2