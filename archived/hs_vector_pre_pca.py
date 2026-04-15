import argparse
import os

import torch
from base_class import ModelDiff
from utils import set_seed


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_name", default="gsm8k", type=str)
	parser.add_argument("--data_dir", default="./data", type=str)
	parser.add_argument("--sub_model", default="gpt-4", type=str)
	parser.add_argument("--add_model", default="gpt-4", type=str)
	parser.add_argument("--prompt_type", default="tool-integrated", type=str)
	parser.add_argument("--num_samples", default=-1, type=int)  # -1 for full data
	parser.add_argument("--alpha", default=0.0, type=float)
	parser.add_argument("--hs_diff_save_path", default=None, type=str)
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--temperature", default=0, type=float)
	parser.add_argument("--n_sampling", default=1, type=int)
	parser.add_argument("--top_p", default=1, type=float)
	parser.add_argument("--max_new_tokens", default=512, type=int)
	parser.add_argument("--shuffle", action="store_true")
	parser.add_argument("--do_sample", type=bool, default=False)
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--split", default="train", type=str)
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--ignore_stop_words", action="store_true")

	args = parser.parse_args()
	args.top_p = (
		1 if args.temperature == 0 else args.top_p
	)  # top_p must be 1 when using greedy sampling (vllm)
	return args


def setup(args):
	if not os.path.exists(os.path.dirname(args.hs_diff_save_path)):
		os.makedirs(os.path.dirname(args.hs_diff_save_path), exist_ok=True)
	elif os.path.exists(args.hs_diff_save_path) and not args.overwrite:
		print("Output file already exists. Skipping.")
		return

	model_diff = ModelDiff()

	prompt_pairs = args.prompt_type.split(",")
	dataset_prompts_pair = []
	for prompt in prompt_pairs:
		samples, _, _ = model_diff.create_response_prompt_and_format(
			args, args.data_name, prompt, reverse=True
		)
		dataset_prompts_pair.append(samples)
	model_diff.generate_hs_pca_pre_diff(
		args, dataset_prompts_pair, args.hs_diff_save_path
	)


if __name__ == "__main__":
	args = parse_args()
	set_seed(args.seed)
	setup(args)
	torch.cuda.empty_cache()
