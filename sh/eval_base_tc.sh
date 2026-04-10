#!/bin/bash
#
#SBATCH --job-name=base_evals
#SBATCH --account=llms-lab
#SBATCH --output=logs/base_evals_%j.out
#SBATCH --error=logs/base_evals_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres-flags=enforce-binding

source ~/activate_conda.sh transfer_env >/dev/null 2>&1
# source ~/activate_conda.sh rlvr_train >/dev/null 2>&1
which python
set -x

BASE_MODELS=(\
	# meta-llama/Llama-3.2-1B \
	# meta-llama/Llama-3.2-3B \
	# meta-llama/Llama-3.2-1B-Instruct \
	# meta-llama/Llama-3.2-3B-Instruct \
	# microsoft/Orca-2-13b \
	# microsoft/Orca-2-7b \
	# Qwen/Qwen2.5-14B \
	# PRIME-RL/Eurus-2-7B-PRIME \
	# Qwen/Qwen2.5-14B-Instruct \
	# allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-0425-1B \
	# allenai/OLMo-2-0425-1B-Instruct \
	# allenai/OLMo-2-1124-13B \
	# allenai/OLMo-2-1124-13B-Instruct \
	# google/gemma-2-9b \
	# google/gemma-2-9b-it \
	# google/gemma-2-2b \
	# google/gemma-2-2b-it \
	# Qwen/Qwen2.5-14B \
	# Qwen/Qwen2.5-14B-Instruct \
	# Qwen/Qwen2.5-7B \
	# Qwen/Qwen2.5-7B-Instruct \
	# nvidia/OpenReasoning-Nemotron-7B \
	# nvidia/DLER-Llama-Nemotron-8B-Merge-Research \
	# nvidia/DLER-R1-7B-Research \
	# nvidia/DLER-R1-1.5B-Research \
	# nvidia/OpenReasoning-Nemotron-1.5B \
	# Qwen/Qwen2.5-Math-1.5B \
	# nvidia/OpenReasoning-Nemotron-14B \
	# allenai/Olmo-3.1-32B-Think \
	# allenai/Olmo-3-1125-32B \
	# allenai/Olmo-3-7B-Think \
	# allenai/Olmo-3-1025-7B \
	# Qwen/Qwen3-8B-Base \
	# Qwen/Qwen3-8B \
	# Qwen/Qwen3-8B-Thinking \
	# Qwen/Qwen3-8B-Base-Thinking \
	# Qwen/Qwen3-4B-Thinking-2507 \
	# Qwen/Qwen3-4B-Base \
	# Qwen/Qwen3-4B \
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
	# Qwen/Qwen1.5-14B \
	# Qwen/Qwen1.5-7B-Chat \
	# Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2.5-1.5B-Instruct \
	# nvidia/DLER-R1-1.5B-Research \
	# Qwen/Qwen3-4B \
	# Qwen/Qwen3-14B \
	# Qwen/Qwen3-14B-Thinking \
	# nvidia/Nemotron-Cascade-8B \
	# meta-llama/Llama-3.1-8B \
	# meta-llama/Llama-3.1-8B-Instruct \
	# meta-llama/Meta-Llama-3-8B \
	# meta-llama/Meta-Llama-3-8B-Instruct \
	# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
	# mistralai/Ministral-3-14B-Base-2512 \
	# mistralai/Ministral-3-14B-Instruct-2512-BF16 \
	mistralai/Ministral-3-8B-Base-2512 \
	# mistralai/Ministral-3-8B-Instruct-2512-BF16 \
	# mistralai/Ministral-3-3B-Base-2512 \
	# mistralai/Ministral-3-3B-Instruct-2512-BF16 \
)
TEST_DATA=(\
	# "math,gsm8k,svamp" \
	# "math,gsm8k,svamp" \
	# "mmlu_stem" \
	# "math,gsm8k,svamp" \
	# "math,gsm8k,svamp" \
	# "mmlu_stem" \
	# "agieval_math,minerva_math,olympiadbench,math500,deepmind_math" \
	"agieval_math,minerva_math" \
	"olympiadbench,deepmind_math" \
	# "math500,deepmind_math" \
	# "agieval_math" \
	# "mmlu_stem" \
	# "mmlu_pro" \
	# "gsm8k"
)
PROMPT_TYPES=(\
	# "general-direct" \
	# "general-cot" \
	# "mcq-direct" \
	# "mcq-cot" \
	# "general-cot" \
	# "general-cot" \
	"general-cot-2" \
	"general-cot-2" \
	# "mcq-cot" \
	# "mcq-cot" \
	# "general-cot-with-demos"
)
NUM_TOKENS=(\
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 2048 \
	# 4096 \
	4096 \
	4096 \
)


OUTPUT_ROOT="/projects/llms-lab/transfer_compare/base_evals_test_prompt_2"
SPLIT="test"
NUM_TEST_SAMPLE=-1
export TOKENIZERS_PARALLELISM=false

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
			# --ignore_stop_words
			# --overwrite
	done
done
