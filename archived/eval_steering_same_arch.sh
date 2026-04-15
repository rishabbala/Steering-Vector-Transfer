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
	# Qwen/Qwen2.5-0.5B \
	# Qwen/Qwen2.5-0.5B \
	Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2.5-7B \
)
FT_MODELS=(\
	# hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo \
	# hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo \
	hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
	# hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
	# hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
)
PROMPT_TYPES=(\
	abel \
	# mathstral \
	# abel \
	# mathstral \
	# qwen-boxed \
)
DATA_TYPES=(\
	"college_math" \
)
# ,minerva_math,olympiadbenc

TASK_TYPES=(\
	reasoning\
)

OUTPUT_ROOT="/projects/llms-lab/math_eval"
SPLIT="test"
NUM_TEST_SAMPLE=200
TOKENIZERS_PARALLELISM=false

for i in "${!BASE_MODELS[@]}"; do
	base="${BASE_MODELS[$i]}"
	ft="${FT_MODELS[$i]}"
	prompt="${PROMPT_TYPES[$i]}"

	for j in "${!DATA_TYPES[@]}"; do
		data="${DATA_TYPES[$j]}"
		task="${TASK_TYPES[$j]}"
		outdir="${OUTPUT_ROOT}/${base##*/}/${task}"
		mkdir -p "${outdir}/logs"
		reasoning_vector_file="${OUTPUT_ROOT}/${base##*/}/${prompt}_${task}.json"

		echo "=== RUNNING ${base}, ${ft}, transferred-${base} ==="

		python3 -u math_eval.py \
			--model_name_or_path ${base} \
			--data_names ${data} \
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
		| tee "${outdir}/logs/${base##*/}_${task}_run_${i}.log"

		python3 -u math_eval.py \
			--model_name_or_path ${ft} \
			--data_name ${data} \
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
		| tee "${outdir}/logs/${ft##*/}_${task}_run_${i}.log"

		# python3 -u math_eval.py \
		# 	--model_name_or_path ${base} \
		# 	--data_name ${data} \
		# 	--output_dir ${outdir} \
		# 	--split ${SPLIT} \
		# 	--prompt_type ${prompt} \
		# 	--num_test_sample ${NUM_TEST_SAMPLE} \
		# 	--rv_file ${reasoning_vector_file} \
		# 	--seed 0 \
		# 	--temperature 0 \
		# 	--n_sampling 1 \
		# 	--top_p 1 \
		# 	--start 0 \
		# 	--end -1 \
		# 	--save_outputs \
		# 	--overwrite \
		# 	--transfer True \
		# | tee "${outdir}/logs/rvec_${base##*/}_${task}_run_${i}.log"
	done
done