#!/bin/bash
#SBATCH --job-name=pytorchjob
#SBATCH --partition=cml-dpart
#SBATCH --account=cml-furongh
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --qos=cml-high_long
#SBATCH --time=36:00:00
#SBATCH --output=/cmlscratch/pan/umd-slurm/logs/outFile-%A_%a.txt
#SBATCH --error=/cmlscratch/pan/umd-slurm/logs/errorFile-%A_%a.txt
#SBATCH --array=1-2




source ~/.bashrc


sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh
sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh >&2
echo "------"
echo "------" >&2
eval $(sed -n "${SLURM_ARRAY_TASK_ID}p" < /cmlscratch/pan/umd-slurm/list-of-commands.sh)



echo "END of SLURM commands"
echo "END of SLURM commands" >&2