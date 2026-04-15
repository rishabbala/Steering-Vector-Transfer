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

BASE_MODELS=(\
	Qwen/Qwen2.5-7B \
	# Qwen/Qwen2.5-7B \
	Qwen/Qwen2.5-7B \
	# Qwen/Qwen2.5-7B \
)
STUDENT_MODELS=(\
	Qwen/Qwen2.5-1.5B \
	# Qwen/Qwen2-1.5B \
	Qwen/Qwen2.5-0.5B \
	# Qwen/Qwen2-0.5B \
)
TEACHER_MODELS=(\
	Qwen/Qwen2.5-1.5B-Instruct \
	# Qwen/Qwen2-1.5B-Instruct \
	Qwen/Qwen2.5-0.5B-Instruct \
	# Qwen/Qwen2-0.5B-Instruct \
)
PROMPT_TYPES=(\
	qwen-boxed \
	# qwen-boxed \
	qwen-boxed \
	# qwen-boxed \
)
DATA_TYPES=(\
	"gsm8k,gaokao2023en,minerva_math" \
)
TASK_TYPES=(\
	reasoning\
)
ALPHAS=(\
	0.5 \
	1 \
	2 \
)
DIRECTION=(\
	"fixed" \
	"increasing" \
	"decreasing" \
)
USE_PARAMS=(\
	# "model.norm.weight" \
	# # "embed" \
	# "embed,lm_head" \
	# "q_proj,k_proj" \
	# "v_proj,o_proj" \
	# "q_proj,k_proj,v_proj,o_proj" \
	# "q_proj,k_proj,v_proj,o_proj,post_attention_layernorm" \
	# # "embed,q_proj,k_proj,v_proj,o_proj,post_attention_layernorm" \
	# "embed,q_proj,k_proj,v_proj,o_proj,post_attention_layernorm,lm_head" \
	# "embed,q_proj,k_proj,v_proj,o_proj,post_attention_layernorm,lm_head,model.norm.weight" \
	"up_proj,gate_proj,down_proj" \
	"input_layernorm,up_proj,gate_proj,down_proj" \
	# "embed,input_layernorm,up_proj,gate_proj,down_proj" \
	"embed,input_layernorm,up_proj,gate_proj,down_proj,lm_head" \
	"embed,input_layernorm,up_proj,gate_proj,down_proj,lm_head,model.norm.weight" \
	"embed,q_proj,k_proj,v_proj,o_proj,post_attention_layernorm,input_layernorm,up_proj,gate_proj,down_proj,lm_head,model.norm.weight" \
)

OUTPUT_ROOT="/projects/llms-lab/math_eval_arch2"
SPLIT="test"
NUM_TEST_SAMPLE=100
TOKENIZERS_PARALLELISM=false

for use_param in "${!USE_PARAMS[@]}"; do
	use_param="${USE_PARAMS[$use_param]}"
	for direction in "${!DIRECTION[@]}"; do
		direction="${DIRECTION[$direction]}"
		for alpha in "${!ALPHAS[@]}"; do
			alpha="${ALPHAS[$alpha]}"
			echo "=== RUNNING weight transfer ${alpha} ==="
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
					weight_diff_save_path="${OUTPUT_ROOT}/${student##*/}/${task}/weight_diff_${prompt}_teacher_${teacher##*/}.pth"

					echo "=== RUNNING weight transfer ${base}, ${student}, ${teacher}, ${prompt}, ${data}, ${task} ==="

					python3 -u weight_vec_arch.py \
						--data_names ${data} \
						--weight_diff_save_path ${weight_diff_save_path} \
						--base_model_name_or_path ${base} \
						--student_model_name_or_path ${student} \
						--teacher_model_name_or_path ${teacher} \
						--alpha_w ${alpha} \
						--use_params ${use_param} \
						--direction ${direction} \
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