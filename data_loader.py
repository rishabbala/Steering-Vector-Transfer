import os
import random
import re

from datasets import load_dataset
from utils import load_jsonl


def load_data(data_name, split, data_dir="./data"):
	if "/" in data_name:
		examples = list(load_jsonl(data_name))

		# add 'idx' in the first column
		if "idx" not in examples[0]:
			examples = [{"idx": i, **example} for i, example in enumerate(examples)]

		# dedepulicate & sort
		examples = sorted(examples, key=lambda x: x["idx"])
		return examples

	data_file = f"{data_dir}/{data_name}"

	print("----------------------------")
	print(data_file, split)
	if os.path.exists(data_file + f"/{split}.jsonl"):
		examples = list(load_jsonl(data_file + f"/{split}.jsonl"))
	elif os.path.exists(data_file + "/test.jsonl"):
		examples = list(load_jsonl(data_file + "/test.jsonl"))
	else:
		if data_name == "gpqa":
			if split == "test":
				data_file = f"{data_dir}/{data_name}/test.jsonl"
				examples = load_dataset(
					"Idavidrein/gpqa",
					"gpqa_diamond",
					split="train",
					# cache_dir=f"{data_dir}/temp",
				)
			else:
				data_file = f"{data_dir}/{data_name}/train.jsonl"
				examples = load_dataset(
					"Idavidrein/gpqa",
					"gpqa_main",
					split="train",
					# cache_dir=f"{data_dir}/temp",
				)

			def preprocess(example):
				choices = [
					re.sub(r"\n", "", ex)
					for ex in [
						example["Correct Answer"],
						example["Incorrect Answer 1"],
						example["Incorrect Answer 2"],
						example["Incorrect Answer 3"],
					]
				]
				correct_answer = choices[0]
				# choices = [
				# 	example["Correct Answer"].strip(),
				# 	example["Incorrect Answer 1"].strip(),
				# 	example["Incorrect Answer 2"].strip(),
				# 	example["Incorrect Answer 3"].strip(),
				# ]
				random.shuffle(choices)
				correct_answer_index = choices.index(correct_answer)

				out_doc = {
					"question": example["Question"].strip()
					+ "\n\nChoices:\n"
					+ "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)),
					"answer": f"({chr(65 + correct_answer_index)})",
					"explanation": example["Explanation"],
				}

				return out_doc

			examples = examples.map(preprocess)
			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
					"explanation",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)


		elif data_name == "xquad_th":
			examples = load_dataset(
				"google/xquad",
				"xquad.th",
				split="validation",
				# cache_dir=f"{data_dir}/temp",
			)

			def preprocess(example):
				out_doc = {
					"question": f"{example['context'].strip()}\n{example['question'].strip()}",
					"answer": example['answers']['text'],
				}

				return out_doc

			examples = examples.map(preprocess)

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)


		elif data_name == "xquad_vi":
			examples = load_dataset(
				"google/xquad",
				"xquad.vi",
				split="validation",
				# cache_dir=f"{data_dir}/temp",
			)

			def preprocess(example):
				out_doc = {
					"question": f"{example['context'].strip()}\n{example['question'].strip()}",
					"answer": example['answers']['text'],
				}

				return out_doc

			examples = examples.map(preprocess)

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)



		elif data_name == "commonsense_qa":
			examples = load_dataset(
				"ChilleD/CommonsenseQA",
				split=split,
				# cache_dir=f"{data_dir}/temp",
			)

			def preprocess(example):
				choices = [
					re.sub(r"\n", "", ex.strip()) for ex in example["choices"]["text"]
				]
				out_doc = {
					"question": (
						example["question"]
						+ ("?" if not example["question"].endswith("?") else "")
					).strip()
					+ "\n\nChoices:\n"
					+ "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)),
					"answer": example["answerKey"],
				}

				return out_doc

			examples = examples.map(preprocess)
			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)

		elif data_name == "svamp":
			examples = load_dataset(
				"ChilleD/SVAMP",
				split=split,
			)

			examples = examples.rename_column("question_concat", "question")
			examples = examples.rename_column("Answer", "answer")

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)

		elif data_name == "deepmind_math":
			examples = load_dataset(
				"di-zhang-fdu/DeepMind_Mathematics_QA",
				split="train",
			)

		elif data_name == "arc_c":
			examples = load_dataset(
				"allenai/ai2_arc",
				"ARC-Challenge",
				split=split,
				# cache_dir=f"{data_dir}/temp",
			)

			def preprocess(example):
				choices = example["choices"]["text"]
				answer = example["answerKey"]

				out_doc = {
					"question": example["question"].strip()
					+ "\n\nChoices:\n"
					+ "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)),
					"answer": f"{answer}",
				}

				return out_doc

			examples = examples.map(preprocess)

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)

		elif data_name == "math500":
			data_file = f"{data_dir}/{data_name}/test.jsonl"
			examples = load_dataset(
				"HuggingFaceH4/MATH-500",
				split="test",
				# cache_dir=f"{data_dir}/temp",
			)

			examples = examples.rename_column("problem", "question")

		elif "strategyqa" in data_name:
			data_file = f"{data_dir}/{data_name}/{split}.jsonl"
			examples = load_dataset(
				"ChilleD/StrategyQA",
				split=split,
			)
			print(len(examples))

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)

		elif "mgsm" in data_name:
			data_file = f"{data_dir}/{data_name}/test.jsonl"
			lang = data_name.split("_")[1]
			examples = load_dataset(
				"juletxara/mgsm",
				lang,
				split="test",
				# cache_dir=f"{data_dir}/temp",
			)

			examples = examples.rename_column("answer", "gt_cot")
			examples = examples.rename_column("answer_number", "answer")

		elif "pop_qa" in data_name:
			data_file = f"{data_dir}/{data_name}/test.jsonl"
			examples = load_dataset(
				"akariasai/PopQA",
				# cache_dir=f"{data_dir}/temp",
			)["test"]

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"possible_answers",
				]:
					to_remove.append(col)

			print(to_remove)
			examples = examples.remove_columns(to_remove)

		elif "winogrande" in data_name:
			data_file = f"{data_dir}/{data_name}/test.jsonl"
			examples = load_dataset(
				"allenai/winogrande",
				"winogrande_debiased",
				split="validation",
				# cache_dir=f"{data_dir}/temp",
			)

			def preprocess(example):
				choices = [example["option1"], example["option2"]]
				answer = example["answer"]

				ab = "AB"
				out_doc = {
					"question": example["sentence"].strip()
					+ "\n\nChoices:\n"
					+ "\n".join(f"({ab[i]}) {c}" for i, c in enumerate(choices)),
					"answer": f"{ab[int(answer) - 1]}",
					"choices": [f"({ab[i]}) {c}" for i, c in enumerate(choices)],
				}

				return out_doc

			examples = examples.map(preprocess)

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
					"choices",
				]:
					to_remove.append(col)

			examples = examples.remove_columns(to_remove)

		elif "agieval_math" in data_name:
			data_file = f"{data_dir}/{data_name}/test.jsonl"
			examples = load_dataset(
				"hails/agieval-math",
				split="test",
				# cache_dir=f"{data_dir}/temp",
			)

			def preprocess(example):
				question = example["query"].split("Q: ")[1].split("A: ")[0].strip()
				out_doc = {
					"question": question,
				}

				return out_doc

			examples = examples.map(preprocess)

			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
				]:
					to_remove.append(col)

			examples = examples.remove_columns(to_remove)

		elif "mmlu_pro" in data_name:
			data_file = f"{data_dir}/{data_name}/test.jsonl"
			examples = load_dataset(
				"TIGER-Lab/MMLU-Pro",
				split="test",
			)

			topics = {
				"math",
				"physics",
				"chemistry",
				"engineering",
				"biology",
				"compute science",
			}  # example set of categories
			examples = examples.filter(lambda ex: ex["category"] in topics)

			print(len(examples))

			def preprocess(example):
				choices = [re.sub(r"\n", "", ex) for ex in example["options"]]
				out_doc = {
					"question": example["question"].strip()
					+ "\n\nChoices:\n"
					+ "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)),
					"answer": f"{(example['answer'])}",
				}

				return out_doc

			examples = examples.map(preprocess)
			to_remove = []
			for col in examples.column_names:
				if col not in [
					"question",
					"answer",
					"cot_content",
					"category",
				]:
					to_remove.append(col)
			examples = examples.remove_columns(to_remove)

		# if data_name == "math":
		#     dataset = load_dataset(
		#         "competition_math",
		#         split=split,
		#         name="main",
		#         cache_dir=f"{data_dir}/temp",
		#     )
		# elif data_name == "gsm8k":
		#     dataset = load_dataset(data_name, split=split)
		# elif data_name == "svamp":
		#     # evaluate on training set + test set
		#     dataset = load_dataset("ChilleD/SVAMP", split="train")
		#     dataset = concatenate_datasets(
		#         [dataset, load_dataset("ChilleD/SVAMP", split="test")]
		#     )
		# elif data_name == "asdiv":
		#     dataset = load_dataset("EleutherAI/asdiv", split="validation")
		#     dataset = dataset.filter(
		#         lambda x: ";" not in x["answer"]
		#     )  # remove multi-answer examples
		# elif data_name == "mawps":
		#     examples = []
		#     # four sub-tasks
		#     for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
		#         sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
		#         for example in sub_examples:
		#             example["type"] = data_name
		#         examples.extend(sub_examples)
		#     dataset = Dataset.from_list(examples)
		elif data_name == "mmlu_stem":
			if split == "test":
				data_file = f"{data_dir}/{data_name}/test.jsonl"
				examples = load_dataset(
					"cais/mmlu", "all", split="test", download_mode="force_redownload"
				)
			else:
				data_file = f"{data_dir}/{data_name}/train.jsonl"
				examples = load_dataset(
					"cais/mmlu", "all", split="validation", download_mode="force_redownload"
				)
			# only keep stem subjects

			def preprocess(example):
				abcd = "ABCD"
				choices = example["choices"]
				answer = abcd[example["answer"]]

				out_doc = {
					"question": example["question"].strip()
					+ "\n\nChoices:\n"
					+ "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)),
					"answer": f"{answer}",
				}

				return out_doc

			stem_subjects = [
				"abstract_algebra",
				"astronomy",
				"college_biology",
				"college_chemistry",
				"college_computer_science",
				"college_mathematics",
				"college_physics",
				"computer_security",
				"conceptual_physics",
				"electrical_engineering",
				"elementary_mathematics",
				"high_school_biology",
				"high_school_chemistry",
				"high_school_computer_science",
				"high_school_mathematics",
				"high_school_physics",
				"high_school_statistics",
				"machine_learning",
			]
			examples = examples.rename_column("subject", "type")
			examples = examples.filter(lambda x: x["type"] in stem_subjects)

			examples = examples.map(preprocess)
		# elif data_name == "carp_en":
		#     dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
		else:
			raise NotImplementedError(data_name)

		os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
		examples.to_json(data_file)

	# add 'idx' in the first column
	if "idx" not in examples[0]:
		examples = [{"idx": i, **example} for i, example in enumerate(examples)]

	# dedepulicate & sort
	examples = sorted(examples, key=lambda x: x["idx"])
	return examples


if __name__ == "__main__":
	examples = load_data("gpqa", "test")
	print(examples[0])
