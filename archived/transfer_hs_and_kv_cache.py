import argparse
import os

from base_class import ModelDiff
from utils import set_seed
import json
import torch


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_names", default="gsm8k,math", type=str)
	parser.add_argument("--data_dir", default="./data", type=str)
	parser.add_argument("--weight_diff_save_path", default=None, type=str)
	parser.add_argument("--hs_diff_save_path", default=None, type=str)
	parser.add_argument("--kv_cache_diff_save_path", default=None, type=str)
	parser.add_argument("--base_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--student_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--teacher_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--alpha_h", default="1.0", type=float)
	parser.add_argument("--c_key", default="1.0", type=float)
	parser.add_argument("--c_value", default="1.0", type=float)
	parser.add_argument("--output_dir", default="./output", type=str)
	parser.add_argument("--prompt_type", default="tool-integrated", type=str)
	parser.add_argument("--split", default="test", type=str)
	parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--temperature", default=0, type=float)
	parser.add_argument("--n_sampling", default=1, type=int)
	parser.add_argument("--top_p", default=1, type=float)
	parser.add_argument("--max_tokens_per_call", default=512, type=int)
	parser.add_argument("--shuffle", action="store_true")
	parser.add_argument("--do_sample", type=bool, default=False)
	parser.add_argument("--batch_size", default=128, type=int)

	args = parser.parse_args()
	args.top_p = (
		1 if args.temperature == 0 else args.top_p
	)  # top_p must be 1 when using greedy sampling (vllm)
	return args


def setup(args):
	model_diff = ModelDiff()
	with open(args.hs_diff_save_path, "r") as f:
		reasoning_vector = json.load(f)["reasoning_vector"]
	reasoning_vector = torch.FloatTensor(reasoning_vector).half().cuda()

	model_diff.load_model_tokenizer(
		args.base_model_name_or_path, reasoning_vector, transfer=True, alpha=args.alpha_h
	)

	# infer & eval
	data_list = args.data_names.split(",")
	results = []
	for data_name in data_list:
		out_file = create_filename(args, data_name)
		results.append(model_diff.add_kv_cache_diff(args, data_name, out_file, load_model=False))

	# add "avg" result to data_list and results
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
	out_file_prefix += "hs_&_kv_cache_"
	out_file_prefix += f"h_alpha_{args.alpha_h}_c_key_{args.c_key}_c_value_{args.c_value}_"
	out_file_prefix += f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
	output_dir = args.output_dir
	out_file = f"{output_dir}/{data_name}/{out_file_prefix}.jsonl"
	os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

	return out_file


if __name__ == "__main__":
	args = parse_args()
	set_seed(args.seed)
	setup(args)
