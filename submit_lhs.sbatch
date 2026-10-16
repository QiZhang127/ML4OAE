#!/bin/bash
#SBATCH -A prio_ey239
#SBATCH --job-name=lhs_oae
#SBATCH --partition=priority_gpu
#SBATCH --gpus=h100:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --chdir=/nfs/roberts/pi/pi_ey239/qz334/ML4OAE
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --exclude=a1127u05n01

set -euo pipefail
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
module --force purge
module load Julia/1.12.4-linux-x86_64
module load OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0
export JULIA_DEPOT_PATH=/nfs/roberts/pi/pi_ey239/qz334/julia_depot

project_dir=/nfs/roberts/pi/pi_ey239/qz334/NumericalOcean
script_dir=/nfs/roberts/pi/pi_ey239/qz334/ML4OAE
row_number=${SLURM_ARRAY_TASK_ID:-${1:-1}}

julia --project="$project_dir" --threads=4 --gcthreads=4,1 \
    "$script_dir/run_lhs_control.jl" "$row_number"
