import argparse
import json
import os
import random
import time
from datetime import datetime

import torch
import torch.nn as nn
from data_loader import load_data
from evaluate import evaluate
from model_utils import generate_completions, load_hf_lm_and_tokenizer
from parser import *
from python_executor import PythonExecutor
from tqdm import tqdm
from trajectory import *
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils import construct_prompt, load_jsonl, save_jsonl, set_seed


class SimpleNet(nn.Module):
	def __init__(
		self,
		input_dim: int = -1,
		output_dim: int = -1,
		num_layers: int = 0,
		bias: bool = True,
	):
		super().__init__()
		if num_layers == 0:
			dims = [input_dim, output_dim]
		else:
			dims = torch.linspace(input_dim, output_dim, num_layers + 2).tolist()
		self.layers = nn.ModuleList()
		for i in range(1, len(dims)):
			self.layers.append(
				nn.Linear(int(dims[i - 1]), int(dims[i]), dtype=torch.bfloat16)
			)
			if i != len(dims) - 1:
				self.layers.append(nn.SiLU())

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for layer in self.layers:
			x = layer(x)
		return x


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_names", default="gsm8k,math", type=str)
	parser.add_argument("--data_dir", default="./data", type=str)
	parser.add_argument("--base_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--transfer_from_model_name_or_path", default="gpt-4", type=str)
	parser.add_argument("--output_dir", default="./output", type=str)
	parser.add_argument("--rv_file", default=None, type=str)
	parser.add_argument("--prompt_type", default="tool-integrated", type=str)
	parser.add_argument("--split", default="test", type=str)
	parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--start", default=0, type=int)
	parser.add_argument("--end", default=-1, type=int)
	parser.add_argument("--temperature", default=0, type=float)
	parser.add_argument("--n_sampling", default=1, type=int)
	parser.add_argument("--top_p", default=1, type=float)
	parser.add_argument("--max_tokens_per_call", default=512, type=int)
	parser.add_argument("--shuffle", action="store_true")
	parser.add_argument("--use_vllm", action="store_true")
	parser.add_argument("--save_outputs", action="store_true")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--use_safetensors", action="store_true")
	parser.add_argument("--transfer", default=False, type=bool)
	parser.add_argument("--transfer_type", default=None, type=str)
	parser.add_argument("--num_shots", type=int, default=0)
	parser.add_argument("--num_layers", type=int, default=20)
	parser.add_argument("--do_sample", type=bool, default=False)
	parser.add_argument("--checkpoints_path", type=str, default=None)
	parser.add_argument(
		"--apply_chat_template",
		action="store_true",
		help="Apply chat template to prompt.",
	)
	parser.add_argument("--pipeline_parallel_size", type=int, default=1)
	parser.add_argument(
		"--adapt_few_shot",
		action="store_true",
		help="Few shot for multiple-choice questions, zero shot for others.",
	)
	args = parser.parse_args()
	args.top_p = (
		1 if args.temperature == 0 else args.top_p
	)  # top_p must be 1 when using greedy sampling (vllm)
	return args


def prepare_data(data_name, args):
	examples = load_data(data_name, args.split, args.data_dir)

	# sample `num_test_sample` from dataset
	if args.num_test_sample > 0:
		# examples = random.sample(examples, min(args.num_test_sample, len(examples)))
		examples = examples[: args.num_test_sample]

	# shuffle
	if args.shuffle:
		random.seed(datetime.now().timestamp())
		random.shuffle(examples)

	# select start and end
	examples = examples[args.start : len(examples) if args.end == -1 else args.end]

	# get out_file name
	out_file_prefix = f"lin_layers_{args.num_layers}_{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
	output_dir = args.output_dir
	out_file = f"{output_dir}/{data_name}/{out_file_prefix}.jsonl"
	os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

	# load all processed samples
	processed_samples = []
	if not args.overwrite:
		processed_files = [
			f
			for f in os.listdir(f"{output_dir}/{data_name}/")
			if f.endswith(".jsonl") and f.startswith(out_file_prefix)
		]
		for f in processed_files:
			processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

	# dedepulicate
	processed_samples = {sample["idx"]: sample for sample in processed_samples}
	processed_idxs = list(processed_samples.keys())
	processed_samples = list(processed_samples.values())
	examples = [example for example in examples if example["idx"] not in processed_idxs]
	return examples, processed_samples, out_file


def setup(args):
	# if args.use_vllm:
	# 	ModelRegistry.register_model("Qwen2ForCausalLMwithReasoningVector", "models.qwen2.qwen2_vllm:Qwen2ForCausalLM")
	# 	print("Loaded Model")

	# load model
	available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
	reasoning_vector = None
	if args.transfer:
		base_config = AutoConfig.from_pretrained(
			args.base_model_name_or_path, trust_remote_code=True
		)
		transfer_config = AutoConfig.from_pretrained(
			args.transfer_from_model_name_or_path, trust_remote_code=True
		)

		linear_model = SimpleNet(
			input_dim=transfer_config.hidden_size,
			output_dim=base_config.hidden_size,
			num_layers=args.num_layers,
		).cuda()

		with open(args.rv_file, "r") as f:
			reasoning_vector = json.load(f)["reasoning_vector"]
		reasoning_vector = torch.FloatTensor(reasoning_vector).bfloat16().cuda()

		## Reasoning vector will be only transferred for  subset of layers, rest will have 0
		# print(tf_model_name, base_model_name, os.listdir("/projects/llms-lab/linear_map_ckpts"))
		reasoning_vector_transferred = (
			torch.zeros((base_config.num_hidden_layers, base_config.hidden_size))
			.bfloat16()
			.cuda()
		)

		# if args.transfer_type == "sparse":
		# 	for i in range(reasoning_vector.shape[0]):
		# 		####### SPARSE VECTOR
		# 		for item in os.listdir(args.checkpoints_path):
		# 			if f"Lfrom_{i}_to_" in item:
		# 				to_layer = int(item.split("_")[-1].replace(".pth", ""))
		# 				print(f"Found Linear Transfer from {i} to {to_layer}", item)
		# 				ckpt = torch.load(f"{args.checkpoints_path}/{item}")
		# 				linear_model.load_state_dict(ckpt['model_state_dict'])
		# 				reasoning_vector_transferred[to_layer] = linear_model(reasoning_vector[i]).detach().cpu()
		# 				break

		if args.transfer_type == "dense":
			for i in range(reasoning_vector_transferred.shape[0]):  #
				########## DENSE VECTOR
				for item in os.listdir(args.checkpoints_path):
					if f"_to_{i}.pth" in item:
						from_layer = int(item.split("_")[1])
						print(f"Found Linear Transfer from {from_layer} to {i}", item)
						ckpt = torch.load(f"{args.checkpoints_path}/{item}")
						linear_model.load_state_dict(ckpt["model_state_dict"])
						rnorm = torch.norm(reasoning_vector[from_layer], dim=-1)
						reasoning_vector_transferred[i] = (
							(
								linear_model(
									reasoning_vector[from_layer]
									/ torch.norm(reasoning_vector[from_layer], dim=-1)
								)
								* rnorm
								* ckpt["ratio"]
							)
							.detach()
							.cpu()
						)

						# reasoning_vector_transferred[i] = (linear_model(reasoning_vector[from_layer])).detach().cpu() # * (r_norm * ckpt['ratio'])
						# reasoning_vector_transferred[i] = reasoning_vector_transferred[i] / torch.norm(reasoning_vector_transferred[i], dim=-1)

						break

		reasoning_vector = reasoning_vector_transferred.detach().cpu()
		del reasoning_vector_transferred, linear_model

		# # Write the data to the config, as vllm doesnt take custom params
		# with open(os.path.join(args.base_model_name_or_path, "config.json"), 'r') as f:
		# 	config_dict = json.load(f)

		# config_dict['transfer_layer'] = True
		# config_dict['reasoning_vector'] = reasoning_vector

		# with open(os.path.join(args.base_model_name_or_path, "config.json"), 'w') as f:
		# 	json.dump(config_dict, f, indent=4)

		# del base_config, transfer_config

	# if args.use_vllm:
	# 	## workaround not being able to parallelize qwen 0.5B
	# 	try:
	# 		# print(args.base_model_name_or_path)
	# 		# model_cls = ModelRegistry.get_supported_archs()
	# 		# print(model_cls)
	# 		# exit()
	# 		llm = LLM(
	# 			model=args.base_model_name_or_path,
	# 			tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
	# 			pipeline_parallel_size=args.pipeline_parallel_size,
	# 			trust_remote_code=True
	# 		)
	# 		print("-------------------- Fail -----------------")
	# 	except:
	# 		print("-------------------- Retry -----------------")
	# 		print(args.base_model_name_or_path)
	# 		llm = LLM(
	# 			model=args.base_model_name_or_path,
	# 			tensor_parallel_size=2,
	# 			pipeline_parallel_size=args.pipeline_parallel_size,
	# 			trust_remote_code=True,
	# 		)
	# 	tokenizer = None
	# 	if args.apply_chat_template:
	# 		tokenizer = AutoTokenizer.from_pretrained(
	# 			args.base_model_name_or_path, trust_remote_code=True
	# 		)
	# else:
	llm, tokenizer = load_hf_lm_and_tokenizer(
		model_name_or_path=args.base_model_name_or_path,
		load_in_half=True,
		use_fast_tokenizer=True,
		use_safetensors=args.use_safetensors,
		reasoning_vector=reasoning_vector,
		transfer=args.transfer,
	)

	# infer & eval
	data_list = args.data_names.split(",")
	results = []
	for data_name in data_list:
		results.append(main(llm, tokenizer, data_name, args))

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

	# if args.transfer:
	# 	# Write the data to the config, as vllm doesnt take custom params
	# 	config_path = os.path.join(args.base_model_name_or_path, "config.json")
	# 	with open(config_path, 'r') as f:
	# 		config_dict = json.load(f)

	# 	config_dict['transfer_layer'] = False
	# 	config_dict['reasoning_vector'] = None

	# 	with open(config_path, 'w') as f:
	# 		json.dump(config_dict, f, indent=4)


def is_multi_choice(answer):
	for c in answer:
		if c not in ["A", "B", "C", "D", "E"]:
			return False
	return True


def main(llm, tokenizer, data_name, args):
	examples, processed_samples, out_file = prepare_data(data_name, args)
	print("=" * 50)
	print("data:", data_name, " ,remain samples:", len(examples))
	if len(examples) > 0:
		print(examples[0])

	executor = None
	# # init python executor
	# if "pal" in args.prompt_type:
	# 	executor = PythonExecutor(get_answer_expr="solution()")
	# else:
	# 	executor = PythonExecutor(get_answer_from_stdout=True)

	samples = []
	for example in tqdm(examples, total=len(examples)):
		idx = example["idx"]

		# parse question and answer
		example["question"] = parse_question(example, data_name)
		if example["question"] == "":
			continue
		gt_cot, gt_ans = parse_ground_truth(example, data_name)
		example["gt_ans"] = gt_ans
		full_prompt = construct_prompt(example, data_name, args)

		if idx == args.start:
			print(full_prompt)

		sample = {
			"idx": idx,
			"question": example["question"],
			"gt_cot": gt_cot,
			"gt": gt_ans,
			"prompt": full_prompt,
		}

		# add remain fields
		for key in [
			"level",
			"type",
			"unit",
			"solution_type",
			"choices",
			"solution",
			"ques_type",
			"ans_type",
			"answer_type",
			"dataset",
			"subfield",
			"filed",
			"theorem",
			"answer",
		]:
			if key in example:
				sample[key] = example[key]
		samples.append(sample)

	# repeat n times
	input_prompts = [
		sample["prompt"] for sample in samples for _ in range(args.n_sampling)
	]
	if args.apply_chat_template:
		input_prompts = [
			tokenizer.apply_chat_template(
				[{"role": "user", "content": prompt.strip()}],
				tokenize=False,
				add_generation_prompt=True,
			)
			for prompt in input_prompts
		]
	remain_prompts = input_prompts
	remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
	end_prompts = []

	max_func_call = (
		1 if args.prompt_type in ["basic-math-cot", "cot", "pal", "mathstral"] else 4
	)

	stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

	if args.prompt_type in ["cot", "abel", "qwen-boxed", "basic-math-cot", "mathstral"]:
		stop_words.extend(
			[
				"assistant",
				"user",
				"_end",
				"_start",
				"Question:",
				"Question",
				"Continuation",
				"Continuing",
				"Continues",
				"You are an AI assistant",
				"[Question]",
				"A:",
				"B:",
				"C:",
				"D:",
				"```",
				"Answer:",
				"Q:",
				"Q.",
				"check if our answer",
			]
		)  # "Solution:""
	if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
		stop_words.extend(["\n\n---", "```output"])
	elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
		stop_words.extend(["Instruction", "Response"])
	elif "jiuzhang" in args.prompt_type:
		stop_words.append("\n\n## Question")
	elif "numina" in args.prompt_type:
		stop_words.append("\n### Problem")
	elif "pure" in args.prompt_type:
		stop_words.append("\n\n\n")

	# start inference
	# measure time use
	start_time = time.time()

	for epoch in range(max_func_call):
		print("-" * 20, "Epoch", epoch)
		current_prompts = remain_prompts
		if len(current_prompts) == 0:
			break

		# get all outputs
		prompts = [item[1] for item in current_prompts]
		# if args.use_vllm:
		# 	if args.transfer:
		# 		# use bsz of 1 to make adding reasoning vector easier
		# 		# obviously slower than auto-batching, but safe rv handling
		# 		# hard to compute the position to add the reasoning vector as vllm uses continuous batching. Less efficient but correct
		# 		outputs = []
		# 		for p in tqdm(prompts):
		# 			out = llm.generate(
		# 				p,
		# 				SamplingParams(
		# 					seed=args.seed,
		# 					temperature=args.temperature,
		# 					top_p=args.top_p,
		# 					max_tokens=args.max_tokens_per_call,
		# 					n=1,
		# 					stop=stop_words,
		# 					stop_token_ids=(
		# 						[151645, 151643]
		# 						if "qwen2" in args.base_model_name_or_path.lower()
		# 						else None
		# 					),
		# 				),
		# 			)
		# 			out = [output.outputs[0].text for output in out]
		# 			outputs.extend(out)
		# 	else:
		# 		outputs = llm.generate(
		# 			prompts,
		# 			SamplingParams(
		# 				seed=args.seed,
		# 				temperature=args.temperature,
		# 				top_p=args.top_p,
		# 				max_tokens=args.max_tokens_per_call,
		# 				n=1,
		# 				stop=stop_words,
		# 				stop_token_ids=(
		# 					[151645, 151643]
		# 					if "qwen2" in args.base_model_name_or_path.lower()
		# 					else None
		# 				),
		# 			),
		# 		)

		# 		outputs = sorted(
		# 			outputs, key=lambda x: int(x.request_id)
		# 		)  # sort outputs by request_id
		# 		outputs = [output.outputs[0].text for output in outputs]
		# else:
		outputs = generate_completions(
			model=llm,
			tokenizer=tokenizer,
			prompts=prompts,
			max_new_tokens=args.max_tokens_per_call,
			batch_size=128,
			stop_id_sequences=stop_words,
			do_sample=args.do_sample,
		)

		assert len(outputs) == len(current_prompts)

		# process all outputs
		remain_prompts = []
		remain_codes = []
		for (i, query), output in zip(current_prompts, outputs):
			output = output.rstrip()
			query += output
			if args.prompt_type == "pal":
				remain_prompts.append((i, query))
				if "```python" in output:
					output = extract_program(query)
				remain_codes.append(output)
			elif args.prompt_type == "cot" or args.prompt_type == "basic-math-cot":
				end_prompts.append((i, query))
			# elif "boxed" not in output and output.endswith("```"):
			# 	program = extract_program(query)
			# 	remain_prompts.append((i, query))
			# 	remain_codes.append(program)
			else:
				end_prompts.append((i, query))

		# execute the remain prompts
		# remain_results = executor.batch_apply(remain_codes)
		# for k in range(len(remain_prompts)):
		# 	i, query = remain_prompts[k]
		# 	res, report = remain_results[k]
		# 	exec_result = res if res else report
		# 	if "pal" in args.prompt_type:
		# 		exec_result = "\\boxed{" + exec_result + "}"
		# 	exec_result = f"\n```output\n{exec_result}\n```\n"
		# 	query += exec_result
		# 	# not end
		# 	if epoch == max_func_call - 1:
		# 		query += "\nReach max function call limit."
		# 	remain_prompts[k] = (i, query)

	# unsolved samples
	print("Unsolved samples:", len(remain_prompts))
	end_prompts.extend(remain_prompts)
	# sort by idx
	end_prompts = sorted(end_prompts, key=lambda x: x[0])

	# remove input_prompt from end_prompt
	codes = []
	assert len(input_prompts) == len(end_prompts)
	for i in range(len(input_prompts)):
		_, end_prompt = end_prompts[i]
		code = end_prompt.split(input_prompts[i])[-1].strip()
		for stop_word in stop_words:
			if stop_word in code:
				code = code.split(stop_word)[0].strip()
		codes.append(code)

	# extract preds
	results = [
		run_execute(executor, code, args.prompt_type, data_name) for code in codes
	]
	time_use = time.time() - start_time

	# put results back to examples
	all_samples = []
	for i, sample in enumerate(samples):
		code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
		result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
		preds = [item[0] for item in result]
		reports = [item[1] for item in result]

		############# NO MULTIPLE CHOICE SO SKIP FOR NOW
		# for j in range(len(preds)):
		# 	if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
		# 		"A",
		# 		"B",
		# 		"C",
		# 		"D",
		# 		"E",
		# 	]:
		# 		preds[j] = choice_answer_clean(code[j])
		# 	elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
		# 		# remove any non-choice char
		# 		preds[j] = "".join(
		# 			[c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
		# 		)

		sample.pop("prompt")
		sample.update({"code": code, "pred": preds, "report": reports})
		all_samples.append(sample)

	# add processed samples
	all_samples.extend(processed_samples)
	all_samples, result_json = evaluate(
		samples=all_samples,
		data_name=data_name,
		prompt_type=args.prompt_type,
		execute=True,
	)

	# save outputs
	if len(processed_samples) < len(all_samples) and args.save_outputs:
		save_jsonl(all_samples, out_file)

	result_json["time_use_in_second"] = time_use
	result_json["time_use_in_minite"] = (
		f"{int(time_use // 60)}:{int(time_use % 60):02d}"
	)

	with open(
		out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
	) as f:
		json.dump(result_json, f, indent=4)
	return result_json


if __name__ == "__main__":
	args = parse_args()
	set_seed(args.seed)
	setup(args)
