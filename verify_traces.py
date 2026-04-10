import glob
import json
import os

from evaluate import evaluate
from parser import extract_answer
from utils import save_jsonl

RERUN_TAG = "rerun - 30:05"


def readJsonFile(file_path):
	code_content = []
	with open(file_path, "r") as file:
		for line in file:
			code_content.append(json.loads(line))
	return code_content


def shouldProcessFile(file_path):
	"""
	Process only if the corresponding metrics file is missing OR
	its time_use_in_second is not equal to the rerun tag.
	"""
	metrics_path = file_path.replace(".jsonl", "_metrics.json")
	if not os.path.isfile(metrics_path):
		return True
	try:
		with open(metrics_path, "r") as f:
			metrics = json.load(f)
		return metrics.get("time_use_in_second") != RERUN_TAG
	except Exception:
		return True


def processFile(file_path):
	dataset = file_path.split("/")[-2]
	if any(x in dataset for x in ["mmlu"]):  # "gsm8k", "svamp",
		return
	prompt_type = file_path.split("/")[-1].split("_")[-1].split(".")[0]
	data = readJsonFile(file_path)
	all_samples = []

	for d in data:
		pred = extract_answer(d["code"][0], dataset, d)
		d.update({"code": d["code"], "pred": [pred], "report": None})
		del d["score"]
		all_samples.append(d)
	all_samples, result_json = evaluate(
		data_name=dataset,
		prompt_type=prompt_type,
		samples=all_samples,
		execute=True,
	)
	save_jsonl(all_samples, file_path)
	result_json["time_use_in_second"] = RERUN_TAG
	with open(file_path.replace(".jsonl", "_metrics.json"), "w") as f:
		json.dump(result_json, f, indent=4)


base_dir = [
	# "/projects/llms-lab/transfer_compare",
	"/projects/llms-lab/transfer_compare/base_evals_test/Qwen3-4B",
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_avg",
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca",
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_ID_avg",
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_ID_pca",
	# "/projects/llms-lab/transfer_compare/base_evals_test/"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_post_training_ID_avg_grid_search",
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_post_training_ID_pca_grid_search"
	# "/projects/llms-lab/transfer_compare/base_evals_test/Qwen2.5-7B"
]


for dir in base_dir:
	json_files = glob.glob(os.path.join(dir, "**", "*.jsonl"), recursive=True)
	for file_path in json_files:
		# if "Ministral" not in file_path:
		# 	continue
		if not shouldProcessFile(file_path):
			print(f"Metrics file already verified: {file_path}")
			continue
		print(f"Processing: {file_path}")
		try:
			processFile(file_path)
			print("Done processing")
		except Exception as e:
			print(f"Error processing: {file_path}")
			print(e)
			pass
