#!/bin/bash
#
#SBATCH --job-name=transfer
#SBATCH --account=llms-lab 
#SBATCH --output=logs/transfer%j.out
#SBATCH --error=logs/transfer%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

bash /home/rishbb/Qwen2.5-Math/sh/isolate_rvec.sh
bash /home/rishbb/Qwen2.5-Math/sh/eval_steering_same_arch.sh
bash /home/rishbb/Qwen2.5-Math/sh/train_lin.sh
bash /home/rishbb/Qwen2.5-Math/sh/eval_transfer.sh