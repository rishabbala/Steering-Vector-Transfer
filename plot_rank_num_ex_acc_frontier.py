import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from scipy.interpolate import griddata

# Global plot style: increase all text sizes for readability
fm.fontManager.addfont(os.path.expanduser("~/.fonts/JuliaMono-Regular.ttf"))
fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Regular.ttf"))
fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Bold.ttf"))

plt.rcParams.update(
	{
		"text.usetex": False,
		"font.family": "monospace",
		"font.monospace": ["Noto Sans Mono", "JuliaMono", "DejaVu Sans Mono"],
		"mathtext.fontset": "dejavusans",
		"figure.dpi": 500,
		"savefig.dpi": 500,
		"axes.titlesize": 50,
		"axes.titleweight": "bold",
		"axes.labelsize": 45,
		"xtick.labelsize": 40,
		"axes.labelweight": "bold",
		"ytick.labelsize": 40,
		"legend.fontsize": 37,
		"hatch.linewidth": 0.5,
	}
)


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


# small caps unicode map
def to_small_caps(text):
	small_caps = {
		"a": "ᴀ",
		"b": "ʙ",
		"c": "ᴄ",
		"d": "ᴅ",
		"e": "ᴇ",
		"f": "ꜰ",
		"g": "ɢ",
		"h": "ʜ",
		"i": "ɪ",
		"j": "ᴊ",
		"k": "ᴋ",
		"l": "ʟ",
		"m": "ᴍ",
		"n": "ɴ",
		"o": "ᴏ",
		"p": "ᴘ",
		"q": "ǫ",
		"r": "ʀ",
		"s": "ꜱ",
		"t": "ᴛ",
		"u": "ᴜ",
		"v": "ᴠ",
		"w": "ᴡ",
		"x": "x",
		"y": "ʏ",
		"z": "ᴢ",
	}
	return "".join(small_caps.get(c.lower(), c) for c in text)


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


def getBestAlphaPerRankAndSamplesPerDatasetAndModel(base_dir):
	data_by_model_dataset = {}

	for filepath in Path(base_dir).rglob("*_metrics.json"):
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

		with open(filepath, "r") as file:
			data = json.load(file)
			accuracy = data.get("acc", 0)

		if model_name not in data_by_model_dataset:
			data_by_model_dataset[model_name] = {}

		if dataset not in data_by_model_dataset[model_name]:
			data_by_model_dataset[model_name][dataset] = {}

		key = (num_samples, rank)
		if key not in data_by_model_dataset[model_name][dataset]:
			data_by_model_dataset[model_name][dataset][key] = []

		data_by_model_dataset[model_name][dataset][key].append((alpha, accuracy))

	best_per_model_dataset = {}
	for model_name, data_by_dataset in data_by_model_dataset.items():
		best_per_model_dataset[model_name] = {}
		for dataset, data_by_samples_rank in data_by_dataset.items():
			best_per_samples_rank = {}
			for (num_samples, rank), alpha_acc_list in data_by_samples_rank.items():
				best_alpha, best_acc = max(alpha_acc_list, key=lambda x: x[1])
				if num_samples not in best_per_samples_rank:
					best_per_samples_rank[num_samples] = {}
				best_per_samples_rank[num_samples][rank] = (best_alpha, best_acc)
			best_per_model_dataset[model_name][dataset] = best_per_samples_rank

	return best_per_model_dataset


def plotTopologicalMap(best_per_samples_rank, dataset_name, model_name):
	ranks_list = []
	samples_list = []
	accuracies_list = []

	for num_samples, rank_dict in best_per_samples_rank.items():
		for rank, (alpha, accuracy) in rank_dict.items():
			ranks_list.append(rank)
			samples_list.append(num_samples)
			accuracies_list.append(accuracy)

	if len(ranks_list) == 0:
		print(f"No data for model {model_name}, dataset {dataset_name}")
		return

	ranks_array = np.array(ranks_list)
	samples_array = np.array(samples_list)
	accuracies_array = np.array(accuracies_list)

	log_ranks = np.log2(ranks_array)
	log_samples = np.log2(samples_array)

	log_rank_min, log_rank_max = log_ranks.min(), log_ranks.max()
	log_samples_min, log_samples_max = log_samples.min(), log_samples.max()

	grid_log_rank, grid_log_samples = np.mgrid[
		log_rank_min:log_rank_max:1000j, log_samples_min:log_samples_max:1000j
	]

	# rbf = Rbf(log_ranks, log_samples, accuracies_array, smooth=0.5)
	# grid_accuracy = rbf(grid_log_rank, grid_log_samples)

	grid_accuracy = griddata(
		(log_ranks, log_samples),
		accuracies_array,
		(grid_log_rank, grid_log_samples),
		method="cubic",
	)

	mask = grid_log_rank > grid_log_samples
	grid_accuracy_masked = np.ma.masked_where(mask, grid_accuracy)

	# Invert log2 grid back to linear space for plotting on log-scaled axes
	grid_rank = 2**grid_log_rank
	grid_samples = 2**grid_log_samples

	# Square figure so x and y have equal physical size
	fig, ax = plt.subplots(figsize=(15, 15))

	contour_filled = ax.contourf(
		grid_rank,
		grid_samples,
		grid_accuracy_masked,
		levels=20,
		cmap="viridis",
		alpha=1.0,
	)
	contour_lines = ax.contour(
		grid_rank,
		grid_samples,
		grid_accuracy_masked,
		levels=10,
		colors="black",
		alpha=0.3,
		linewidths=0.5,
	)
	ax.clabel(contour_lines, inline=True, fmt=lambda x: f"{x:.1f}", fontsize=10)

	rank_min, rank_max = ranks_array.min(), ranks_array.max()
	samples_min, samples_max = samples_array.min(), samples_array.max()
	diagonal_x = np.logspace(
		np.log2(max(rank_min, samples_min)),
		np.log2(min(rank_max, samples_max)),
		100,
		base=2,
	)
	ax.fill_between(
		diagonal_x,
		diagonal_x,
		samples_min,
		color="gray",
		alpha=0.3,
		label=r"Invalid region ($k > n$)",
	)

	# ax.scatter(
	# 	ranks_array,
	# 	samples_array,
	# 	c=accuracies_array,
	# 	s=100,
	# 	cmap="viridis",
	# 	edgecolors="black",
	# 	linewidth=1,
	# 	zorder=5,
	# )

	# Use base-2 log axes so ticks match the log2 computations above
	ax.set_xscale("log", base=2)
	ax.set_yscale("log", base=2)
	# Make the axes box square (so x/y plot area is equal), even with log scales.
	try:
		ax.set_box_aspect(1)
	except Exception:
		pass

	# Shorter colorbar to reduce whitespace
	cbar = plt.colorbar(contour_filled, ax=ax, shrink=0.78, pad=0.05, aspect=28)
	cbar.set_label("Accuracy", rotation=270, labelpad=50)
	cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
	# cbar.ax.tick_params(labelsize=20)

	ax.set_xlabel(rf"Rank $\mathbf{{k}}$")
	ax.set_ylabel(rf"Number of Samples $\mathbf{{n}}$")
	m1 = model_name.split("_-_")[-1].split("-")[-1]
	m0 = model_name.split("_+_")[0]

	ax.set_title(
		f"{m0}" + f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;{m1}}}}}$",
		ha="center",
		# fontsize=20,
		pad=50,
		x=0.55,
	)
	# ax.grid(True, alpha=0.3, which="both", color="black")
	ax.tick_params(axis="both", which="both")
	ax.legend(loc="lower right")

	plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.1)
	# plt.tight_layout(pad=0.1)
	plt.savefig(
		f"results/topological_map_{model_name}_{dataset_name}.pdf",
		bbox_inches="tight",
		# pad_inches=0.01,
	)
	plt.show()


base_dir = (
	"/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_cot_avg_grid_search"
)
best_per_model_dataset = getBestAlphaPerRankAndSamplesPerDatasetAndModel(base_dir)

print(f"Found {len(best_per_model_dataset)} models")
if not os.path.exists("results"):
	os.makedirs("results")
for model_name in best_per_model_dataset.keys():
	print(
		f"  Model: {model_name}, Datasets: {list(best_per_model_dataset[model_name].keys())}"
	)

for model_name, best_per_dataset in best_per_model_dataset.items():
	if "OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B" not in model_name:
		continue
	# try:
	for dataset_name, best_per_samples_rank in best_per_dataset.items():
		print(f"\nPlotting model: {model_name}, dataset: {dataset_name}")
		plotTopologicalMap(best_per_samples_rank, dataset_name, model_name)
	# except Exception as e:
	# 	print(f"Error plotting model: {model_name}, dataset: {dataset_name}")
	# 	print(e)
