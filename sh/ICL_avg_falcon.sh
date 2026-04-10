#!/bin/bash
#
#SBATCH --job-name=ICL_avg_falcon
#SBATCH --account=llms-lab
#SBATCH --output=logs/ICL_avg_falcon_%j.out
#SBATCH --error=logs/ICL_avg_falcon_%j.out
#SBATCH --time=23:00:00
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres-flags=enforce-binding


source ~/activate_conda.sh rlvr_train >/dev/null 2>&1
set -x

TEST_SPLIT="test"
NUM_TEST_SAMPLE=-1

huggingface-cli login --token $HF_TOKEN


BASE_MODEL=(\
	# Qwen/Qwen1.5-7B \
	Qwen/Qwen1.5-14B \
	# allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-0425-1B \
	# allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-1124-13B \
)
SUB_CAP_MODEL=(\
	# Qwen/Qwen1.5-14B \
	Qwen/Qwen1.5-7B \
	# allenai/OLMo-2-0425-1B \
	# allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-1124-13B \
	# allenai/OLMo-2-1124-7B \
)
ADD_CAP_MODEL=(\
	# Qwen/Qwen1.5-14B \
	Qwen/Qwen1.5-7B \
	# allenai/OLMo-2-0425-1B \
	# allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-1124-13B \
	# allenai/OLMo-2-1124-7B \
)
TEST_DATA=(\
	"gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "math" \
	# "svamp" \
)
STEERING_DATA=(\
	"gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "math" \
	# "svamp" \
)
PCA_DATA=(\
	"gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "gsm8k" \
	# "math" \
	# "svamp" \
)
PROMPT=(\
	# "general-direct,general-cot" \
	# "general-direct,general-cot" \
	# "general-direct,general-cot" \
    # "general-cot,general-cot-with-demos" \
	# "general-cot,general-cot-with-demos" \
	# "general-cot,general-cot-with-demos" \
	# "general-cot,general-cot-with-demos" \
	# "general-cot,general-cot-with-demos" \
	# "general-cot,general-cot-with-demos" \
	"general-direct,general-cot-with-demos" \
	# "general-direct,general-cot-with-demos" \
	# "general-direct,general-cot-with-demos" \
	# "general-direct,general-cot-with-demos" \
	# "general-direct,general-cot-with-demos" \
	# "general-direct,general-cot-with-demos" \

)
NUM_TEST_TOKENS=(\
	512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
	# 512 \
)
ALPHAS=(\
	0.05 \
	# 0.1 \
	# 0.2 \
	# 0.5 \
)
RANK=(\
	# 1 \
	4 \
	# 16 \
	# 64 \
	# 128 \
	# 256 \
	# 512 \
)
NUM_TRAIN_SAMPLE=64

OUTPUT_ROOT="/projects/llms-lab/transfer_compare//projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_icl_avg_direct,cot+icl/num_train_samples_${NUM_TRAIN_SAMPLE}"
TOKENIZERS_PARALLELISM=false

for i in "${!BASE_MODEL[@]}"; do
	base_model="${BASE_MODEL[$i]}"
	sub_model="${SUB_CAP_MODEL[$i]}"
	add_model="${ADD_CAP_MODEL[$i]}"
	for j in "${!TEST_DATA[@]}"; do
		test_data="${TEST_DATA[$j]}"
		prompt="${PROMPT[$j]}"
		max_tokens="${NUM_TEST_TOKENS[$j]}"
		steering_data="${STEERING_DATA[$j]}"
		pca_data="${PCA_DATA[$j]}"
		steering_datastring=${steering_data}
		pca_datastring=${pca_data}
		for a in "${!ALPHAS[@]}"; do
			alpha="${ALPHAS[$a]}"
			for r in "${!RANK[@]}"; do
				rank="${RANK[$r]}"

				outdir="${OUTPUT_ROOT}/${base_model##*/}_+_${add_model##*/}_-_${sub_model##*/}/rank_${rank}/pca_${pca_datastring}_steering_${steering_datastring}"
				weight_save_path="${OUTPUT_ROOT}/base_${base_model##*/}_sub_${sub_model##*/}_pca_data_${pca_datastring}_rank_${rank}_num_train_samples_${NUM_TRAIN_SAMPLE}.pth"
				hs_diff_save_path="${OUTPUT_ROOT}/sub_${sub_model##*/}_add_${add_model##*/}_steering_data_${steering_datastring}_num_train_samples_${NUM_TRAIN_SAMPLE}.pth"

				echo "=== RUNNING Layerwise Steering Vector Computation ${base_model}, ${sub_model}, ${add_model}, ${steering_data}, ${alpha} ==="

				if [[ -n "${hs_diff_save_path}" ]] && [[ ! -e "${hs_diff_save_path}" ]]; then
					python3 -u hs_vector_simple.py \
						--data_name "${steering_data}" \
						--sub_model "${sub_model}" \
						--add_model "${add_model}" \
						--prompt_type "${prompt}" \
						--num_samples ${NUM_TRAIN_SAMPLE} \
						--split "train" \
						--alpha ${alpha} \
						--hs_diff_save_path "${hs_diff_save_path}" \
						--seed 0 \
						--max_new_tokens 1 \
						--batch_size 4
				fi

				echo "=== RUNNING SVD Computation ${base_model}, ${sub_model}, ${add_model}, ${pca_data}, ${rank} ==="
				if [[ -n "${weight_save_path}" ]] && [[ ! -e "${weight_save_path}" ]]; then
					python3 -u hs_svd.py \
						--data_name "${pca_data}" \
						--base_model "${base_model}" \
						--sub_model "${sub_model}" \
						--add_model "${add_model}" \
						--prompt_type "${prompt}" \
						--split "train" \
						--num_samples ${NUM_TRAIN_SAMPLE} \
						--rank ${rank} \
						--weight_save_path "${weight_save_path}" \
						--max_new_tokens 1 \
						--batch_size 1
				fi


				echo "=== RUNNING Transfer ${base_model}, ${sub_model}, ${add_model}, ${test_data}, ${prompt}, ${alpha}, ${rank} ==="
				python3 -u transfer_hs_arch.py \
					--data_name_list "${test_data}" \
					--base_model "${base_model}" \
					--sub_model "${sub_model}" \
					--add_model "${add_model}" \
					--hs_diff_save_path "${hs_diff_save_path}" \
					--weight_save_path "${weight_save_path}" \
					--alpha ${alpha} \
					--output_dir "${outdir}" \
					--prompt_type "${prompt}" \
					--split "${TEST_SPLIT}" \
					--num_samples ${NUM_TEST_SAMPLE} \
					--seed 0 \
					--max_new_tokens ${max_tokens} \
					--batch_size 16
					# --overwrite
			done
		done
	done
done