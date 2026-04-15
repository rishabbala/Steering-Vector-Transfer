#!/bin/bash
#
#SBATCH --job-name=steering_vector_same_arch
#SBATCH --account=llms-lab 
#SBATCH --output=logs/steering_vector_same_arch%j.out
#SBATCH --error=logs/steering_vector_same_arch%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G


set -ex

export VLLM_CONFIGURE_LOGGING=0
export TORCHDYNAMO_VERBOSE=1
#–– Define your grid of experiments ––
BASE_MODELS=(\
	Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2.5-7B \
	# Qwen/Qwen2.5-7B \
	# Qwen/Qwen2.5-7B \
	# Qwen/Qwen2.5-7B \
)
TRANSFER_MODELS=(\
	Qwen/Qwen2.5-0.5B \
	# Qwen/Qwen2.5-0.5B \
	# Qwen/Qwen2.5-0.5B \
	# Qwen/Qwen2.5-0.5B \
	# Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2.5-1.5B \
)
PROMPT_TYPES_FROM=(\
	abel \
	# qwen-boxed \
	# abel \
	# mathstral \
	# abel \
	# mathstral \
)
PROMPT_TYPES=(\
	abel \
	# mathstral \
	# qwen-boxed \
	# qwen-boxed \
	# qwen-boxed \
	# qwen-boxed \
)
DATA_TYPES=(\
	"gsm8k,college_math,gaokao2023en" \
)
#,minerva_math,olympiadbench

TASK_TYPES=(\
	reasoning\
)
TRANSFER_TYPE=(\
	# "sparse" \
	"dense" \
)
NUM_LIN_LAYERS=(\
	0 \
	# 20 \
	# 100 \
)



OUTPUT_ROOT="/projects/llms-lab/math_eval"
SPLIT="test"
NUM_TEST_SAMPLE=200
TOKENIZERS_PARALLELISM=false

for n in "${!NUM_LIN_LAYERS[@]}"; do
	num_layers=${NUM_LIN_LAYERS[$n]}
	for i in "${!BASE_MODELS[@]}"; do
		base="${BASE_MODELS[$i]}"
		transfer="${TRANSFER_MODELS[$i]}"
		prompt_from="${PROMPT_TYPES_FROM[$i]}"
		prompt="${PROMPT_TYPES[$i]}"

		for t in "${!TRANSFER_TYPE[@]}"; do
			ttype="${TRANSFER_TYPE[$t]}"
			for j in "${!DATA_TYPES[@]}"; do
				outdir="${OUTPUT_ROOT}/${transfer##*/}_to_${base##*/}/${ttype}"
				data="${DATA_TYPES[$j]}"
				task="${TASK_TYPES[$j]}"
				reasoning_vector_file="${OUTPUT_ROOT}/${transfer##*/}/${prompt_from}_${task}.json"
				linear_checkpoints_pth="${OUTPUT_ROOT}/${transfer##*/}_to_${base##*/}/linear_checkpoints_${ttype}_from_${prompt_from}_to_${prompt}/${task}/${num_layers}"

				mkdir -p "${outdir}/logs"
				echo "=== RUNNING Transfer ${transfer} -> ${base} ==="

				TOKENIZERS_PARALLELISM=false \
				python3 -u transfer.py \
					--base_model_name_or_path ${base} \
					--transfer_from_model_name_or_path ${transfer} \
					--data_name ${data} \
					--rv_file ${reasoning_vector_file} \
					--output_dir ${outdir} \
					--split ${SPLIT} \
					--prompt_type ${prompt} \
					--num_test_sample ${NUM_TEST_SAMPLE} \
					--seed 0 \
					--temperature 0 \
					--n_sampling 1 \
					--top_p 1 \
					--start 0 \
					--end -1 \
					--save_outputs \
					--overwrite \
					--transfer True \
					--transfer_type ${ttype} \
					--num_layers ${num_layers} \
					--checkpoints_path ${linear_checkpoints_pth} \
				| tee "${outdir}/logs/transfer_${data}_run_${i}.log"
			done
		done
	done
done