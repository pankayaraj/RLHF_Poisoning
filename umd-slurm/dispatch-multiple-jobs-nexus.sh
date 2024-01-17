#!/bin/bash
#SBATCH --job-name=pytorchjob
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64gb
#SBATCH --qos=scavenger
#SBATCH --time=72:00:00
#SBATCH --output=/cmlscratch/pan/umd-slurm/logs/outFile-%A_%a.txt
#SBATCH --error=/cmlscratch/pan/umd-slurm/logs/errorFile-%A_%a.txt
#SBATCH --array=1-3




source ~/.bashrc



sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh
sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh >&2
echo "------"
echo "------" >&2
eval $(sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh)


echo "END of SLURM commands"
echo "END of SLURM commands" >&2