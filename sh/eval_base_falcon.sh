#!/bin/bash
#
#SBATCH --job-name=base_evals
#SBATCH --account=llms-lab
#SBATCH --output=logs/base_evals_%j.out
#SBATCH --error=logs/base_evals_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres-flags=enforce-binding

# source ~/activate_conda.sh transfer_env >/dev/null 2>&1
source ~/activate_conda.sh rlvr_train >/dev/null 2>&1
which python
set -x

BASE_MODELS=(\
	# microsoft/Orca-2-13b \
	# microsoft/Orca-2-7b \
	# Qwen/Qwen2.5-14B \
	# PRIME-RL/Eurus-2-7B-PRIME \
	# Qwen/Qwen2.5-14B-Instruct \
	# allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-1124-13B \
	# allenai/OLMo-2-1124-13B-Instruct \
	# google/gemma-2-9b \
	# google/gemma-2-9b-it \
	# google/gemma-2-2b \
	# Qwen/Qwen2.5-14B \
	# Qwen/Qwen2.5-14B-Instruct \
	# Qwen/Qwen2.5-7B \
	# Qwen/Qwen2.5-7B-Instruct \
	# Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2.5-1.5B-Instruct \
	# nvidia/DLER-R1-7B-Research \
	# nvidia/OpenReasoning-Nemotron-7B \
	# nvidia/DLER-Llama-Nemotron-8B-Merge-Research \
	# meta-llama/Meta-Llama-3-8B \
	# meta-llama/Llama-3.1-8B \
	# google/gemma-3-12b-pt \
	# google/gemma-3-12b-it \
	# google/gemma-3-4b-pt \
	# google/gemma-3-4b-it \
	# EssentialAI/rnj-1 \
	# EssentialAI/rnj-1-instruct \
	# ibm-granite/granite-3.3-8b-base \
	# ibm-granite/granite-3.3-8b-instruct \
	# ibm-granite/granite-3.3-2b-base \
	# ibm-granite/granite-3.3-2b-instruct \
	# nvidia/OpenReasoning-Nemotron-14B \
	# sail/Sailor-7B \
	# Qwen/Qwen1.5-7B \
	# Qwen/Qwen1.5-14B \
	Qwen/Qwen1.5-7B-Chat \
	Qwen/Qwen1.5-14B-Chat \
	# google/gemma-2-9b \
	# google/gemma-2-2b \
	# FreedomIntelligence/Apollo2-2B \
	# FreedomIntelligence/Apollo2-9B \
	# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
	# mistralai/Ministral-3-8B-Base-2512 \
	# mistralai/Ministral-3-8B-Instruct-2512-BF16 \
	# mistralai/Ministral-3-3B-Base-2512 \
	# mistralai/Ministral-3-3B-Instruct-2512-BF16 \
	# Qwen/Qwen3-8B-Base \
	# Qwen/Qwen3-8B \
	# Qwen/Qwen3-8B-Thinking \
	# Qwen/Qwen3-8B-Base-Thinking \
	# Qwen/Qwen3-4B-Thinking-2507 \
	# Qwen/Qwen3-4B-Base \
	# Qwen/Qwen3-4B \
)
TEST_DATA=(\
	# "math,gsm8k,svamp" \
	# "math,gsm8k,svamp" \
	"gsm8k,math" \
	# "mmlu_stem" \
	# "math,gsm8k,svamp" \
	# "math,gsm8k,svamp" \
	# "mmlu_stem" \
	# "agieval_math,minerva_math,olympiadbench,math500,deepmind_math" \
	# "agieval_math,minerva_math" \
	# "olympiadbench,deepmind_math" \
	# "olympiadbench" \
	# "math500,deepmind_math" \
	# "gpqa,mmlu_pro" \
	# "mmlu_stem" \
	# "gpqa" \
	# "xquad_th" \
	# "xquad_vi" \
	# "math" \
	# "gsm8k" \
)
PROMPT_TYPES=(\
	# "general-direct" \
	# "general-cot" \
	# "general-cot-2" \
	# "general-cot-2" \
	"general-cot-with-demos" \
	# "prompt-repetition" \
	# "prompt-repetition-x3-direct" \
	# "prompt-repetition-x3-cot" \
	# "mcq-cot-with-demos" \
	# "mcq-direct-with-demos" \
	# "mcq-cot" \
	# "general-cot" \
	# "general-cot" \
	# "mcq-cot" \
	# "th-cot" \
	# "vi-cot" \
)
NUM_TOKENS=(\
	512 \
	# 512 \
	# 512 \
	# 512 \
	# 1024 \
	# 2048 \
	# 4096 \
	# 4096 \
)


OUTPUT_ROOT="/projects/llms-lab/transfer_compare/base_evals_test"
SPLIT="test"
NUM_TEST_SAMPLE=-1

for i in "${!BASE_MODELS[@]}"; do
	base="${BASE_MODELS[$i]}"
	outdir="${OUTPUT_ROOT}/${base##*/}"

	for j in "${!TEST_DATA[@]}"; do
		prompt="${PROMPT_TYPES[$j]}"
		data="${TEST_DATA[$j]}"
		max_tokens="${NUM_TOKENS[$j]}"

		echo "=== Evaluating model: ${base} with prompt: ${prompt} on data: ${data} ==="

		python3 -u base_evals.py \
			--data_names ${data} \
			--base_model_name_or_path ${base} \
			--output_dir ${outdir} \
			--prompt_type ${prompt} \
			--split ${SPLIT} \
			--num_samples ${NUM_TEST_SAMPLE} \
			--seed 0 \
			--max_new_tokens ${max_tokens} \
			--batch_size 16
			# --overwrite
			# --ignore_stop_words
	done
done
