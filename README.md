# Official Code for [The Master Key Hypothesis: Unlocking Cross-Model Capability Transfer via Linear Subspace Alignment](https://arxiv.org/abs/2604.06377)

The codebase is build on top of [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). Required packages listed in [requirements.txt](https://github.com/rishabbala/Steering-Vector-Transfer/blob/main/requirements.txt).

The code is split into three parts

## Part 1: Master Key Computation

To compute the Maskter Key using mean aggregation run:
```
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
  --batch_size "${bsz}"
```

and for PCA accgregation run:
```
python3 -u hs_vector_post_pca.py \
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
  --batch_size "${bsz}"
```

Arguments:
`steering_data` - The data on which the steering vector is computed \
`sub_model` - The Source Locked Model \
`add_model` - The Source Unlocked Model \
`prompt_type` - A list of two prompts, the first prompt is applied to the Locked models and the second one to the Unlocked models \
`num_samples` - Number of training samples to use. Set to -1 to use all examples \
`split` - The data split on which the steering vector is computed. Always set to `train` \
`hs_diff_save_path` - Path to store the steering vector \
`max_new_tokens` - Set to 1 \
`batch_size` - Batch size for computation \

---

## Part 2: Linear Transformation

For mean aggregator, use:
```
python3 -u hs_svd.py \
  --data_name "${pca_data}" \
  --base_model "${base_model}" \
  --sub_model "${sub_model}" \
  --prompt_type "${prompt}" \
  --split "train" \
  --num_samples ${NUM_TRAIN_SAMPLE} \
  --rank ${rank} \
  --weight_save_path "${weight_save_path}" \
  --max_new_tokens 1 \
  --batch_size 1
```

and for PCA aggregator:
```
python3 -u hs_svd_zero_centered.py \
  --data_name "${pca_data}" \
  --base_model "${base_model}" \
  --sub_model "${sub_model}" \
  --prompt_type "${prompt}" \
  --split "train" \
  --num_samples ${NUM_TRAIN_SAMPLE} \
  --rank ${rank} \
  --weight_save_path "${weight_save_path}" \
  --max_new_tokens 1 \
  --batch_size 1
```

Arguments:
`pca_data` - Data on which the linear transformation is computed. Typically this would be the same as `steering_data` \
`base_model` - The Target Locked model \
`sub_model` - The Source Locked model \
`prompt_type` - A list of two prompts, the first prompt is applied to both the Target Locked and Source Locked models \
`split` - The data split on which the mapping is computed, typically set to `train` \
`num_samples` - Number of samples to use \
`rank` - The rank of the transformation. `rank <= num_samples` \
`weight_save_path` - The location to store the mapping \

---

## Part 3: Test Time Intervention

```
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
```

Arguments:
`data_name_list` - A single dataset or a list of datasets \
`base_model` - The Target Locked model \
`sub_model`  - The Source Locked model \
`add_model`  - The Source Unlocked model \
`hs_diff_save_path`, `weight_save_path` - Corresponding save paths \
`alpha` - Steering strength \
`output_dir` - Directory to save the generations and evaluations \
`prompt_type` - Pair of prompts, the first one is applied to the Target model \
`split` - Set to test \
`num_sample` - Set to -1 to evaluate on all samples

---

A complete example is provided in [sh/CoT_avg_tc.sh](https://github.com/rishabbala/Steering-Vector-Transfer/blob/main/sh/CoT_avg_tc.sh)
