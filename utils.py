import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np
from examples import get_examples


def set_seed(seed: int = 42) -> None:
	np.random.seed(seed)
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
	with open(file, "r", encoding="utf-8") as f:
		for line in f:
			try:
				yield json.loads(line)
			except:
				print("Error in loading:", line)
				exit()


def save_jsonl(samples, save_path):
	# ensure path
	folder = os.path.dirname(save_path)
	os.makedirs(folder, exist_ok=True)

	with open(save_path, "w", encoding="utf-8") as f:
		for sample in samples:
			f.write(json.dumps(sample, ensure_ascii=False) + "\n")
	print("Saved to", save_path)


def lower_keys(example):
	new_example = {}
	for key, value in example.items():
		if key != key.lower():
			new_key = key.lower()
			new_example[new_key] = value
		else:
			new_example[key] = value
	return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots=0):
	if num_shots == 0:
		return []

	# if any(["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"] in data_name):
	# 	data_name = "gsm8k"
	# if any(["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"] in data_name):
	# 	data_name = "math"
	# if any(["sat_math"] in data_name):
	# 	data_name = "mmlu_stem"
	# if any(
	# 	[
	# 		"gaokao2024_I",
	# 		"gaokao2024_II",
	# 		"gaokao_math_qa",
	# 		"gaokao2024_mix",
	# 		"cn_middle_school",
	# 	]
	# 	in data_name
	# ):
	# 	data_name = "gaokao"

	if prompt_type in ["tool-integrated"]:
		prompt_type = "tora"

	# Use all examples
	print("Numshots", num_shots)
	if num_shots == -1:
		return EXAMPLES[data_name]
	else:
		return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
	# "ifeval": (
	# 	"Respond to the following question by strictly following the instructions provided:\n\nQuestion:\n{input}\n\nAnswer:\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	# "qwen-general-chat": (
	# 	"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
	# 	'<|im_start|>user\n{input}\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem.<|im_end|>\n'
	# 	"<|im_start|>assistant\nLet's think step by step.\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	# "olmo-general-chat": (
	# 	"<|endoftext|><|system|>\nYou are a helpful assistant.\n"
	# 	'<|user|>\n{input}\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem.\n'
	# 	"<|assistant|>\nLet's think step by step.\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	# "llama-3-2-general-chat": (
	# 	"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n"
	# 	'<|start_header_id|>user<|end_header_id|>\n\n{input}\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem.<|eot_id|>\n'
	# 	"<|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	# "qwen-mcq-chat": (
	# 	"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
	# 	'<|im_start|>user\n{input}\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem. If none of the choices are correct, select the choice that is closest to the correct answer.<|im_end|>\n'
	# 	"<|im_start|>assistant\nLet's think step by step.\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	# "olmo-mcq-chat": (
	# 	"<|endoftext|><|system|>\nYou are a helpful assistant.\n"
	# 	'<|user|>\n{input}\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem. If none of the choices are correct, select the choice that is closest to the correct answer.\n'
	# 	"<|assistant|>\nLet's think step by step.\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	# "llama-3-2-mcq-chat": (
	# 	"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n"
	# 	'<|start_header_id|>user<|end_header_id|>\n\n{input}\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem. If none of the choices are correct, select the choice that is closest to the correct answer.<|eot_id|>\n'
	# 	"<|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	# "phi3-general-chat": (
	# 	"<|user|>\nPlease reason step by step, and end your answer with \"The final answer is <atok> [answer] </atok>.\" where [answer] is the response to the problem.\n{input}\n<|end|>\n"
	# 	"<|assistant|>\nLet's think step by step.\n",
	# 	"{output}",
	# 	"\n\n",
	# ),
	"orca-general-chat-cot": (
		f'<|im_start|>user\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem.\nQuestion:\n{input}\n<|im_end|>\n<|im_start|>assistant\nLet\'s think step by step.\n',
		"{output}",
		"\n\n",
	),
	"orca-general-chat-direct": (
		f'<|im_start|>user\nSolve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\nQuestion:\n{input}\n<|im_end|>\n<|im_start|>assistant\nLet\'s think step by step.\n',
		"{output}",
		"\n\n",
	),
	"deepseek-r1-distill-qwen-general-chat-cot": (
		"<｜begin▁of▁sentence｜>"
		'<｜User｜>\nPlease reason step by step, and end your answer with "The final answer is <atok> [answer] </atok>." where [answer] is the response to the problem.{input}\n'
		"<｜Assistant｜><think>\nLet's think step by step.\n",
		"{output}",
		"\n\n",
	),
	"general-cot": (
		'Reason step by step and give a final answer to the following question. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nAnswer: Let\'s think step by step.\n',
		"{output}",
		"\n\n",
	),
	"general-cot-2": (
		"Place the final answer within \\boxed{{}} at the end of your response.\n\nQuestion:\n{input}\n\nAnswer: Let's think step by step.\n",
		"{output}",
		"\n\n",
	),
	"th-cot": (
		'ให้คิดอย่างเป็นขั้นตอนและให้คำตอบสุดท้ายสำหรับคำถามต่อไปนี้ คำตอบของคุณควรลงท้ายด้วย "คำตอบสุดท้ายคือ [คำตอบ]" โดยที่ [คำตอบ] คือคำตอบที่ถูกต้องของปัญหา\n\nคำถาม:\n{input}\n\nคำตอบ: มาคิดทีละขั้นตอนกันเถอะ\n',
		"{output}",
		"\n\n",
	),
	"vi-cot": (
		'H̄ı̂ khid xỳāng pĕn k̄hận txn læa h̄ı̂ khả txb s̄udtĥāy s̄ảh̄rạb khảt̄hām t̀x pị nī̂ khả txb k̄hxng khuṇ khwr lngtĥāy d̂wy"khả txb s̄udtĥāy khụ̄x [khả txb]" doythī̀ [khả txb] khụ̄x khả txb thī̀ t̄hūk t̂xng k̄hxng pạỵh̄ā\n\nkhảt̄hām:\n{input}\n\nkhả txb: Mā khid thī la k̄hận txn kạn t̄hexa\n',
		"{output}",
		"\n\n",
	),
	"strategy-cot": (
		'Reason step by step and give a final answer to the following question. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is either "True" or "False".\n\nQuestion:\n{input}\n\nAnswer: Let\'s think step by step.\n',
		"{output}",
		"\n\n",
	),
	"mcq-cot": (
		'Reason step by step and select the choice that correctly answers the following question. If none of the choices are correct, select the choice that is closest to the correct answer. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct choice to the question.\n\nQuestion:\n{input}\n\nAnswer: Let\'s think step by step.\n',
		"{output}",
		"\n\n",
	),
	"general-direct": (
		'Solve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nAnswer:\n',
		"{output}",
		"\n\n",
	),
	"prompt-repetition-x3-direct": (
		'Solve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nLet me repeat that:\n\nSolve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nLet me repeat that one more time:\n\nSolve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nAnswer:\n',
		"{output}",
		"\n\n",
	),
	"prompt-repetition": (
		'Solve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nSolve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nAnswer:\n',
		"{output}",
		"\n\n",
	),
	"prompt-repetition-x3-cot": (
		'Reason step by step and give a final answer to the following question. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nLet me repeat that:\n\nReason step by step and give a final answer to the following question. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nLet me repeat that one more time:\n\nReason step by step and give a final answer to the following question. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct solution to the problem.\n\nQuestion:\n{input}\n\nAnswer: Let\'s think step by step.\n',
		"{output}",
		"\n\n",
	),
	"th-direct": (
		'จงแก้ปัญหาต่อไปนี้และวางคำตอบไว้ท้ายสุด  คำตอบของคุณควรลงท้ายด้วย "คำตอบสุดท้ายคือ [คำตอบ]" โดยที่ [คำตอบ] คือคำตอบที่ถูกต้องของโจทย์\n\nคำถาม:\n{input}\n\nคำตอบ:\n',
		"{output}",
		"\n\n",
	),
	"strategy-direct": (
		'Solve the following question and place the answer at the end. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is either "True" or "False".\n\nQuestion:\n{input}\n\nAnswer:\n',
		"{output}",
		"\n\n",
	),
	"mcq-direct": (
		'Select the choice that correctly answers the following question and place the answer at the end. If none of the choices are correct, select the choice that is closest to the correct answer. Your response should always end with "The final answer is <atok> [answer] </atok>." where [answer] is the correct choice to the question.\n\nQuestion:\n{input}\n\nAnswer:\n',
		"{output}",
		"\n\n",
	),
	"direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
	"cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
	"basic-math-cot": (
		"Question: {input}\nAnswer: Let's think step by step.\n",
		"{output}",
		"\n\n\n",
	),
	"pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
	"tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
	"self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
	"tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
	"wizard_zs": (
		"### Instruction:\n{input}\n\n### Response: Let's think step by step.",
		"{output}",
		"\n\n\n",
	),
	"platypus_fs": (
		"### Instruction:\n{input}\n\n### Response:\n",
		"{output}",
		"\n\n\n",
	),
	"deepseek-math": (
		"User: {input}\nPlease reason step by step, "
		"and put your final answer within \\boxed{{}}.\n\nAssistant:",
		"{output}",
		"\n\n\n",
	),
	"kpmath": (
		"User: Please reason step by step and put your final answer at the end "
		'with "The answer is: ".\n\n{input}\n\nAssistant:',
		"{output}",
	),
	"jiuzhang": (
		"## Question\n{input}\n\n## Solution\n",
		"{output}",
		"\n\n\n",
	),
	"jiuzhang_tora": (
		"## Question\n{input}\n\n## Code Solution\n",
		"{output}",
		"\n\n\n",
	),
	"jiuzhang_nl": (
		"## Question\n{input}\n\n## Natural Language Solution\n",
		"{output}",
		"\n\n\n",
	),
	"mmiqc": (
		'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
		"{output}",
		"\n\n\n",
	),
	"abel": (
		"Question:\n{input}\nAnswer:\nLet's think step by step.",
		"{output}",
		"\n\n",
	),
	"shepherd": ("{input}\n", "{output}", "\n\n\n"),
	"qwen-boxed": (
		"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
		"<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
		"<|im_start|>assistant\n",
		"{output}",
		"\n\n",
	),
	"qwen25-math-cot": (
		"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
		"<|im_start|>user\n{input}<|im_end|>\n"
		"<|im_start|>assistant\n",
		"{output}",
		"\n\n",
	),
	"mathstral": (
		"{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
		"{output}",
		"\n\n",
	),
	"internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
	"internlm-math-chat": (
		"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
		"{output}",
		"\n\n",
	),
	"mistral": (
		"[INST] {input}[/INST]",
		"{output}",
		"\n\n",
	),
	"numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
}


def construct_prompt(example, data_name, prompt_type, args, use_type=None, num_shots=0):
	# if use_type is None:
	# 	prompt_type = prompt_type
	# elif use_type == "contrastive":
	# 	prompt_type = args.prompt_type_contrastive
	# elif use_type == "base":
	# 	prompt_type = args.prompt_type_base
	# else:
	# 	raise ValueError

	demos = load_prompt(data_name, prompt_type, num_shots)
	prompt_type = prompt_type
	if prompt_type == "platypus_fs":
		prompt_type = "cot"
	if prompt_type == "tool-integrated":
		prompt_type = "tora"

	prompt_temp = PROMPT_TEMPLATES[prompt_type]

	splitter = prompt_temp[2]
	input_template, output_template, splitter = (
		prompt_temp[0],
		prompt_temp[1],
		prompt_temp[2],
	)
	if prompt_type == "ifeval":
		full_prompt = input_template.format(input=example["prompt"]) + splitter
		full_prompt.strip(" ")
		return full_prompt
	if prompt_type == "general-cot":
		# Hotfix to support putting all demos into a single turn
		demo_prompt = splitter.join(
			[
				f"Question:\n{q.strip()}\n\nAnswer: Let's think step by step.\n{a.strip()}"
				for q, a in demos
			]
		)
	elif prompt_type == "gpqa":
		demo_prompt = splitter.join(
			[
				input_template.format(input=q) + output_template.format(output=a)
				for q, a in demos
			]
		)
	else:
		demo_prompt = splitter.join(
			[
				input_template.format(input=q) + output_template.format(output=a)
				for q, a in demos
			]
		)
	context = input_template.format(input=example["question"])
	if len(demo_prompt) == 0:
		full_prompt = context
	else:
		if prompt_type == "general-cot":
			# Hotfix to supportting put all demos into a single turn
			full_prompt = (
				demo_prompt
				+ splitter
				+ f"Question:\n{example['question'].strip()}\n\nAnswer: Let's think step by step.\n"
			)
			full_prompt = input_template.split("\n\n")[0] + "\n\n" + full_prompt
		else:
			full_prompt = demo_prompt + splitter + context

	if prompt_type == "platypus_fs":
		full_prompt_temp = (
			"Below is an instruction that describes a task. "
			"Write a response that appropriately completes the request.\n\n"
			"### Instruction:\n{instruction}\n\n### Response:\n"
		)
		full_prompt = full_prompt_temp.format(instruction=full_prompt)

	if prompt_type == "tora":
		full_prompt = (
			"""Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
			+ full_prompt
		)

	return full_prompt.strip(" ")  # important!


key_map = {
	"gt": "Ground Truth",
	"pred": "Prediction",
	"gt_cot": "Reference CoT",
	"score": "Score",
}


def show_sample(sample, print_all_preds=False):
	print("==" * 20)
	for key in ["idx", "type", "level", "dataset"]:
		if key in sample:
			# capitalize
			print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
	print("Question:", repr(sample["question"]))
	if "code" in sample:
		if print_all_preds:
			for code in sample["code"]:
				print("-" * 20)
				print("code:", code)
			print("Execution:", sample["report"])
		else:
			print("Solution:\n", sample["code"][0])
			print("Execution:", sample["report"][0])
	if "pred" in sample:
		print("Prediction:", repr(sample["pred"][0]))
	for key in ["gt", "score", "unit", "gt_cot"]:
		if key in sample:
			_key = key_map.get(key, key)
			print("{}: {}".format(_key, repr(sample[key])))
	print()
