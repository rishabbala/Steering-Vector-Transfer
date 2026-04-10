#!/bin/bash
#
#SBATCH --job-name=steering_vector_same_arch
#SBATCH --account=llms-lab 
#SBATCH --output=logs/steering_vector%j.out
#SBATCH --error=logs/steering_vector%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G


set -ex

export VLLM_CONFIGURE_LOGGING=0
export TORCHDYNAMO_VERBOSE=1
#–– Define your grid of experiments ––
BASE_MODELS=(\
	Qwen/Qwen2.5-7B \
	meta-llama/Llama-3.1-8B \
)
STUDENT_MODELS=(\
	Qwen/Qwen2-7B \
	meta-llama/Meta-Llama-3-8B \
)
TEACHER_MODELS=(\
	Qwen/Qwen2-7B-Instruct \
	meta-llama/Meta-Llama-3-8B-Instruct \
)
PROMPT_TYPES=(\
	qwen-boxed \
	llama-prompt \
)
DATA_TYPES=(\
	"gsm8k,gaokao2023en,minerva_math,olympiadbench" \
)
TASK_TYPES=(\
	reasoning\
)
ALPHA_H=(\
	0.5 \
	1 \
	2 \
)
C_KEY=(\
	0.5 \
	1 \
	2 \
)
C_VALUE=(\
	0.5 \
	1 \
	2 \
)
OUTPUT_ROOT="/projects/llms-lab/math_eval"
SPLIT="test"
NUM_TEST_SAMPLE=100
TOKENIZERS_PARALLELISM=false

for alpha_h in "${!ALPHA_H[@]}"; do
	alpha_h="${ALPHA_H[$alpha_h]}"
	for c_key in "${!C_KEY[@]}"; do
		c_key="${C_KEY[$c_key]}"
		for c_value in "${!C_VALUE[@]}"; do
			c_value="${C_VALUE[$c_value]}"
			for i in "${!BASE_MODELS[@]}"; do
				base="${BASE_MODELS[$i]}"
				student="${STUDENT_MODELS[$i]}"
				teacher="${TEACHER_MODELS[$i]}"
				prompt="${PROMPT_TYPES[$i]}"
			
				for j in "${!DATA_TYPES[@]}"; do
					data="${DATA_TYPES[$j]}"
					task="${TASK_TYPES[$j]}"
					outdir="${OUTPUT_ROOT}/${base##*/}/${task}"
					mkdir -p "${outdir}/logs"
					hs_diff_save_path="${OUTPUT_ROOT}/${student##*/}/${task}/hs_diff_${prompt}_teacher_${teacher##*/}.pth"
					kv_cache_diff_save_path="${OUTPUT_ROOT}/${student##*/}/${task}/kv_cache_diff_${prompt}_teacher_${teacher##*/}.pth"

					echo "=== RUNNING hidden state transfer ${base}, ${student}, ${teacher}, ${prompt}, ${data}, ${task} ==="

					python3 -u transfer_hs_and_kv_cache.py \
						--data_names ${data} \
						--kv_cache_diff_save_path ${kv_cache_diff_save_path} \
						--hs_diff_save_path ${hs_diff_save_path} \
						--base_model_name_or_path ${base} \
						--student_model_name_or_path ${student} \
						--teacher_model_name_or_path ${teacher} \
						--alpha_h ${alpha_h} \
						--c_key ${c_key} \
						--c_value ${c_value} \
						--output_dir ${outdir} \
						--prompt_type ${prompt} \
						--split ${SPLIT} \
						--num_test_sample ${NUM_TEST_SAMPLE} \
						--seed 0 \
						--temperature 0 \
						--n_sampling 1 \
						--top_p 1 \
						--max_tokens_per_call 2048 \
						--batch_size 128
				done
			done
		done
	done
done
