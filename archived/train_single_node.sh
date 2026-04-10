#!/bin/bash
#SBATCH --job-name=trainer
#SBATCH --account=llms-lab 
#SBATCH --output=logs/trainer%j.out
#SBATCH --error=logs/trainer%j.out
#SBATCH --open-mode=append
#SBATCH --time=00:20:00
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --gres-flags=enforce-binding

set -ex

export VLLM_CONFIGURE_LOGGING=0
export TORCHDYNAMO_VERBOSE=1

# python ./train.py

MACHINE_RANK=0
NUM_PROCESSES=4
echo "Available GPUs:"
nvidia-smi --list-gpus

accelerate launch \
	--config_file ./configs/accelerate_config.yaml \
	--num_machines 1 \
	--num_processes $NUM_PROCESSES \
	--machine_rank $MACHINE_RANK \
	train.py