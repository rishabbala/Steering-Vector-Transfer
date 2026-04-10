import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf, griddata


def extractAlpha(filename):
	# Extract alpha value from filename
	match = re.search(r"alpha_([0-9.]+)", filename)
	if match:
		return float(match.group(1))
	return None


def extractRank(filepath):
	# Extract rank from filepath
	match = re.search(r"rank_(\d+)", filepath)
	if match:
		return int(match.group(1))
	return None


def extractNumTrainSamples(filepath):
	# Extract number of training samples from filepath
	match = re.search(r"num_train_samples_(\d+)", filepath)
	if match:
		return int(match.group(1))
	return None


def extractDataset(filepath):
	# Extract dataset name from filepath pattern: /pca_DATASET_steering_DATASET/DATASET/
	match = re.search(r"/pca_(\w+)_steering_\w+/(\w+)/", filepath)
	if match:
		return match.group(2)
	return None


def extractModelName(filepath):
	# Extract model name pattern: /OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B/
	match = re.search(r"/num_train_samples_\d+/([^/]+)/", filepath)
	if match:
		return match.group(1)
	return None


def getAvgCodeLength(jsonl_file):
	# Calculate average length of code text from jsonl file
	total_length = 0
	count = 0

	with open(jsonl_file, "r") as file:
		for line in file:
			data = json.loads(line)
			if "code" in data and data["code"]:
				for code_text in data["code"]:
					total_length += len(code_text)
					count += 1

	return total_length / count if count > 0 else 0


def getMaxAvgCodeLengthPerRankAndSamplesPerDatasetAndModel(base_dir):
	data_by_model_dataset = {}

	for filepath in Path(base_dir).rglob("*.jsonl"):
		filepath_str = str(filepath)
		rank = extractRank(filepath_str)
		num_samples = extractNumTrainSamples(filepath_str)
		alpha = extractAlpha(filepath.name)
		dataset = extractDataset(filepath_str)
		model_name = extractModelName(filepath_str)

		if (
			rank is None
			or num_samples is None
			or alpha is None
			or dataset is None
			or model_name is None
		):
			continue

		avg_code_length = getAvgCodeLength(filepath)

		if model_name not in data_by_model_dataset:
			data_by_model_dataset[model_name] = {}

		if dataset not in data_by_model_dataset[model_name]:
			data_by_model_dataset[model_name][dataset] = {}

		key = (num_samples, rank)
		if key not in data_by_model_dataset[model_name][dataset]:
			data_by_model_dataset[model_name][dataset][key] = []

		data_by_model_dataset[model_name][dataset][key].append((alpha, avg_code_length))

	max_per_model_dataset = {}
	for model_name, data_by_dataset in data_by_model_dataset.items():
		max_per_model_dataset[model_name] = {}
		for dataset, data_by_samples_rank in data_by_dataset.items():
			max_per_samples_rank = {}
			for (num_samples, rank), alpha_length_list in data_by_samples_rank.items():
				max_alpha, max_length = max(alpha_length_list, key=lambda x: x[1])
				if num_samples not in max_per_samples_rank:
					max_per_samples_rank[num_samples] = {}
				max_per_samples_rank[num_samples][rank] = (max_alpha, max_length)
			max_per_model_dataset[model_name][dataset] = max_per_samples_rank

	return max_per_model_dataset


def plotTopologicalMapCodeLength(max_per_samples_rank, dataset_name, model_name):
	ranks_list = []
	samples_list = []
	lengths_list = []

	for num_samples, rank_dict in max_per_samples_rank.items():
		for rank, (alpha, avg_length) in rank_dict.items():
			ranks_list.append(rank)
			samples_list.append(num_samples)
			lengths_list.append(avg_length)

	if len(ranks_list) == 0:
		print(f"No data for model {model_name}, dataset {dataset_name}")
		return

	ranks_array = np.array(ranks_list)
	samples_array = np.array(samples_list)
	lengths_array = np.array(lengths_list)

	log_ranks = np.log10(ranks_array)
	log_samples = np.log10(samples_array)

	log_rank_min, log_rank_max = log_ranks.min(), log_ranks.max()
	log_samples_min, log_samples_max = log_samples.min(), log_samples.max()

	grid_log_rank, grid_log_samples = np.mgrid[
		log_rank_min:log_rank_max:1000j, log_samples_min:log_samples_max:1000j
	]

	# rbf = Rbf(log_ranks, log_samples, lengths_array, smooth=0.5)
	# grid_length = rbf(grid_log_rank, grid_log_samples)

	grid_length = griddata(
		(log_ranks, log_samples),
		lengths_array,
		(grid_log_rank, grid_log_samples),
		method="cubic",
	)

	mask = grid_log_rank > grid_log_samples
	grid_length_masked = np.ma.masked_where(mask, grid_length)

	grid_rank = 10**grid_log_rank
	grid_samples = 10**grid_log_samples

	fig, ax = plt.subplots(figsize=(12, 8))

	contour_filled = ax.contourf(
		grid_rank,
		grid_samples,
		grid_length_masked,
		levels=20,
		cmap="viridis",
		alpha=0.8,
	)
	contour_lines = ax.contour(
		grid_rank,
		grid_samples,
		grid_length_masked,
		levels=10,
		colors="black",
		alpha=0.3,
		linewidths=0.5,
	)
	ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")

	rank_min, rank_max = ranks_array.min(), ranks_array.max()
	samples_min, samples_max = samples_array.min(), samples_array.max()
	diagonal_x = np.logspace(
		np.log10(max(rank_min, samples_min)), np.log10(min(rank_max, samples_max)), 100
	)
	ax.fill_between(
		diagonal_x,
		diagonal_x,
		samples_min,
		color="gray",
		alpha=0.3,
		label="Invalid region (rank > num_samples)",
	)

	scatter = ax.scatter(
		ranks_array,
		samples_array,
		c=lengths_array,
		s=100,
		cmap="viridis",
		edgecolors="black",
		linewidth=1,
		zorder=5,
	)

	ax.set_xscale("log")
	ax.set_yscale("log")

	cbar = plt.colorbar(contour_filled, ax=ax)
	cbar.set_label("Avg Code Length", rotation=270, labelpad=20)

	ax.set_xlabel("Rank")
	ax.set_ylabel("Number of Training Samples")
	ax.set_title(f"Topological Map: {model_name} - {dataset_name} (Avg Code Length)")
	ax.grid(True, alpha=0.3, which="both")
	ax.legend()

	plt.tight_layout()
	plt.savefig(
		f"results/rank_num_ex_code_length_topological_{model_name}_{dataset_name}.png",
		dpi=300,
	)
	plt.show()


base_dir = "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_cot_pca_model_fam_grid_search/"
max_per_model_dataset = getMaxAvgCodeLengthPerRankAndSamplesPerDatasetAndModel(base_dir)

print(f"Found {len(max_per_model_dataset)} models")
for model_name in max_per_model_dataset.keys():
	print(
		f"  Model: {model_name}, Datasets: {list(max_per_model_dataset[model_name].keys())}"
	)

for model_name, max_per_dataset in max_per_model_dataset.items():
	if "OLMo-2-1124-13B_+_Qwen1.5-7B-Chat_-_Qwen1.5-7B" not in model_name:
		continue
	try:
		for dataset_name, max_per_samples_rank in max_per_dataset.items():
			print(f"\nPlotting model: {model_name}, dataset: {dataset_name}")
			plotTopologicalMapCodeLength(max_per_samples_rank, dataset_name, model_name)
	except Exception as e:
		print(f"Error plotting model: {model_name}, dataset: {dataset_name}")
		print(e)
