import json
import random

data_names = [
	# "agieval_math",
	# "minerva_math",
	# "olympiadbench",
	# "gaokao2023en",
	# "math500",
	# "deepmind_math",
	# "arc_c",
	# "mmlu_stem",
	# "mmlu_pro",
	# "xquad_vi",
	"gpqa",
]


for data_name in data_names:
	with open(f"data/{data_name}/train.jsonl", "r") as f:
		lines = f.readlines()

	total_lines = len(lines)
	random.shuffle(lines)

	with open(f"data/{data_name}/train.jsonl", "w") as f:
		f.writelines(lines)

	print(f"Shuffled {len(lines)} lines in train.jsonl")
