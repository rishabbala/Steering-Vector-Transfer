import json
import random

data_names = [
	# "agieval_math",
	# "minerva_math",
	# "olympiadbench",
	# "gaokao2023en",
	# "math500",
	# "deepmind_math",
	# "mmlu_stem",
	# "mmlu_pro",
	"xquad_vi",
]


for data_name in data_names:
	with open(f"data/{data_name}/test.jsonl", "r") as f:
		lines = f.readlines()

	total_lines = len(lines)
	sample_indices = set(random.sample(range(total_lines), min(500, total_lines)))

	train_lines = []
	test_lines = []

	for idx, line in enumerate(lines):
		if idx in sample_indices:
			train_lines.append(line)
		else:
			test_lines.append(line)

	with open(f"data/{data_name}/train.jsonl", "w") as f:
		f.writelines(train_lines)

	with open(f"data/{data_name}/test.jsonl", "w") as f:
		f.writelines(test_lines)

	print(f"Moved {len(train_lines)} lines to train.jsonl")
	print(f"Kept {len(test_lines)} lines in test.jsonl")
