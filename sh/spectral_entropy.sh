#!/bin/bash
#
#SBATCH --job-name=spectral_entropy
#SBATCH --account=llms-lab
#SBATCH --output=logs/spectral_entropy_%j.out
#SBATCH --error=logs/spectral_entropy_%j.out
#SBATCH --time=15:00:00
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres-flags=enforce-binding

source ~/activate_conda.sh rlvr_train >/dev/null 2>&1
set -x

huggingface-cli login --token $HF_TOKEN


BASE_MODEL=(\
	allenai/OLMo-2-1124-13B \
	# allenai/OLMo-2-1124-7B \
	# Qwen/Qwen1.5-14B \
	# Qwen/Qwen1.5-7B \
)
SUB_CAP_MODEL=(\
	allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-0425-1B \
	# Qwen/Qwen1.5-7B \
	# Qwen/Qwen1.5-14B \
)
ADD_CAP_MODEL=(\
	allenai/OLMo-2-1124-7B \
	# allenai/OLMo-2-0425-1B \
	# Qwen/Qwen1.5-7B \
	# Qwen/Qwen1.5-14B \
)
TEST_DATA=(\
	"gsm8k" \
	"math" \
	# "svamp" \
	# "agieval_math" \
	# "deepmind_math" \
	# "minerva_math" \
	# "olympiadbench" \
)
STEERING_DATA=(\
	"gsm8k" \
	"math" \
	# "svamp" \
	# "agieval_math" \
	# "deepmind_math" \
	# "minerva_math" \
	# "olympiadbench" \
)
PCA_DATA=(\
	"gsm8k" \
	"math" \
	# "svamp" \
	# "agieval_math" \
	# "deepmind_math" \
	# "minerva_math" \
	# "olympiadbench" \
)
PROMPT=(\
	"general-direct,general-cot" \
	"general-direct,general-cot" \
	# "general-direct,general-cot" \
	# "general-cot,general-cot" \
	# "general-cot,general-cot" \
	# "general-cot,general-cot" \
	# "general-cot,general-cot" \
)
NUM_TEST_TOKENS=(\
	512 \
	512 \
	# 512 \
	# 4096 \
)
ALPHAS=(\
	0.05 \
	# 0.1 \
	# 0.2 \
	# 0.5 \
)
RANK=(\
	1 \
	# 4 \
	# 16 \
	# 64 \
	# 128 \
	# 256 \
	# 512 \
)
NUM_TRAIN_SAMPLE=1024

OUTPUT_ROOT="/projects/llms-lab/transfer_compare/spectral_entropy/num_train_samples_${NUM_TRAIN_SAMPLE}"
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

				# if [[ -n "${hs_diff_save_path}" ]] && [[ ! -e "${hs_diff_save_path}" ]] && [[ ${rank} -eq 1 ]]; then
				python3 -u spectral_entropy.py \
					--data_name "${steering_data}" \
					--sub_model "${sub_model}" \
					--add_model "${add_model}" \
					--prompt_type "${prompt}" \
					--num_samples ${NUM_TRAIN_SAMPLE} \
					--split "test" \
					--alpha ${alpha} \
					--hs_diff_save_path "${hs_diff_save_path}" \
					--seed 0 \
					--max_new_tokens 1 \
					--batch_size 4

				# 	python3 -u spectral_entropy.py \
				# 		--data_name "${steering_data}" \
				# 		--base_model "${base_model}" \
				# 		--sub_model "${sub_model}" \
				# 		--add_model "${add_model}" \
				# 		--prompt_type "${prompt}" \
				# 		--num_samples ${NUM_TRAIN_SAMPLE} \
				# 		--split "train" \
				# 		--alpha ${alpha} \
				# 		--hs_diff_save_path "${hs_diff_save_path}" \
				# 		--seed 0 \
				# 		--max_new_tokens 1 \
				# 		--batch_size 4
				# fi

				# echo "=== RUNNING SVD Computation ${base_model}, ${sub_model}, ${add_model}, ${pca_data}, ${rank} ==="
				# if [[ -n "${weight_save_path}" ]] && [[ ! -e "${weight_save_path}" ]]; then
				# 	python3 -u hs_svd.py \
				# 		--data_name "${pca_data}" \
				# 		--base_model "${base_model}" \
				# 		--sub_model "${sub_model}" \
				# 		--add_model "${add_model}" \
				# 		--prompt_type "${prompt}" \
				# 		--split "train" \
				# 		--num_samples ${NUM_TRAIN_SAMPLE} \
				# 		--rank ${rank} \
				# 		--weight_save_path "${weight_save_path}" \
				# 		--max_new_tokens 1 \
				# 		--batch_size 1
				# fi
			done
		done
	done
done