#!/bin/bash

#SBATCH --job-name=trainer_multinode
#SBATCH --account=llms-lab 
#SBATCH --output=logs/trainer_multinode_%j.out
#SBATCH --error=logs/trainer_multinode_%j.out
#SBATCH --open-mode=append
#SBATCH --time=00:20:00
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres-flags=enforce-binding

source ~/.bashrc
hf_token=$(cat $HOME/hf_token.txt)
huggingface-cli login --token $hf_token

export GPUS_PER_NODE=4
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# https://github.com/huggingface/accelerate/issues/2246

# ## Torchrun launcher
# export LAUNCHER=" \
#     torchrun \
#     --nnodes $SLURM_NNODES \
#     --nproc_per_node $GPUS_PER_NODE \
#     --rdzv_id 29500 \
#     --rdzv_backend c10d \
#     "

#     # --rdzv_endpoint $head_node_ip:$UID

## Accelerate Launcher 
export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
	--machine_rank $SLURM_NODEID \
    "

# export SCRIPT="train.py"
# export SCRIPT_ARGS=" \
#     --mixed_precision fp16 \
#     --output_dir ${ACCELERATE_DIR}/examples/output \
#     "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER train.py" 
srun $CMD