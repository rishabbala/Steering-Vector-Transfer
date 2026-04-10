#!/bin/bash
#
#SBATCH --job-name=gemini
#SBATCH --account=llms-lab
#SBATCH --output=logs/gemini_evals_%j.out
#SBATCH --error=logs/gemini_evals_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source ~/activate_conda.sh >/dev/null 2>&1
which python
set -x

python ./verify_gemini.py