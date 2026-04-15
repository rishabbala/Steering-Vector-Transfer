import json
import random


# for data_name in data_names:
with open(f"data/gpqa/train.jsonl", "r") as f:
	train = f.readlines()
with open(f"data/gpqa/test.jsonl", "r") as f:
	test = f.readlines()

new_train = []
for t in train:
	found = False
	q = json.loads(t)["question"].split('\n\n')[0]
	for t2 in test:
		q2 = json.loads(t2)["question"].split('\n\n')[0]
		if q == q2:
			found = True
			break
	if not found:
		new_train.append(t)

with open(f"data/gpqa/train.jsonl", "w") as f:
	f.writelines(new_train)
	print(f"Deduplicated {len(new_train)} lines in train.jsonl")
