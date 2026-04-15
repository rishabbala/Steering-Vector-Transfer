import argparse
import os

import torch
from base_class import ModelDiff
from utils import set_seed


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_names", default="gsm8k,math", type=str)
	parser.add_argument("--data_dir", default="./data", type=str)
	parser.add_argument("--weight_diff_save_path", default=None, type=str)
	parser.add_argument("--base_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--student_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--teacher_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--alpha", default="1.0", type=float)
	parser.add_argument("--use_params", default="all", type=str)
	parser.add_argument("--output_dir", default="./output", type=str)
	parser.add_argument("--prompt_type", default="tool-integrated", type=str)
	parser.add_argument("--split", default="test", type=str)
	parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--temperature", default=0, type=float)
	parser.add_argument("--n_sampling", default=1, type=int)
	parser.add_argument("--top_p", default=1, type=float)
	parser.add_argument("--max_new_tokens", default=512, type=int)
	parser.add_argument("--shuffle", action="store_true")
	parser.add_argument("--do_sample", type=bool, default=False)
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--overwrite", action="store_true")

	args = parser.parse_args()
	args.top_p = (
		1 if args.temperature == 0 else args.top_p
	)  # top_p must be 1 when using greedy sampling (vllm)
	return args


def setup(args):
	model_diff = ModelDiff()
	model_diff.generate_weight_diff(args)
	model_diff.apply_weight_diff(args)

	# infer & eval
	data_list = args.data_names.split(",")
	prompt_list = args.prompt_type.split(",")
	results = []
	for data_name, prompt_type in zip(data_list, prompt_list):
		out_file = create_filename(args, data_name)
		if not os.path.exists(os.path.dirname(out_file)):
			os.makedirs(os.path.dirname(out_file))
		elif not args.overwrite and os.path.exists(out_file):
			continue
		results.append(
			model_diff.generate_response(
				args,
				data_name,
				prompt_type,
				out_file,
			)
		)

	# add "avg" result to data_list and results
	if len(results) == 0:
		print("Evals exist")
		return
	data_list.append("avg")
	results.append(
		{
			"acc": sum([result["acc"] for result in results]) / len(results),
		}
	)

	# print all results
	pad = max([len(data_name) for data_name in data_list])
	print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
	print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))

	return


def create_filename(args, data_name):
	# get out_file name
	model_name = "/".join(args.base_model_name_or_path.split("/")[-2:]).replace(
		"/", "_"
	)
	out_file_prefix = f"{model_name}_"
	out_file_prefix += f"alpha_{args.alpha}_{args.split}_num_samples_{args.num_test_sample}_max_gen_tokens_{args.max_new_tokens}"
	output_dir = args.output_dir
	out_file = f"{output_dir}/{data_name}/{out_file_prefix}.jsonl"

	return out_file


if __name__ == "__main__":
	args = parse_args()
	set_seed(args.seed)
	setup(args)
	torch.cuda.empty_cache()
