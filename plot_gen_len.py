#!/usr/bin/env python3
"""
Generation Length Analysis for Transfer Learning Methods
Compares baseline models (direct and CoT) with transfer methods (avg and pca)
across multiple datasets and model families.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
import matplotlib.font_manager as fm

import os

# Paths to data directories
BASE_DIR = Path("/projects/llms-lab/transfer_compare")
AVG_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_avg"
PCA_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_pca"
BASE_EVAL_DIR = BASE_DIR / "base_evals_test"

# Target datasets
DATASETS = ["gsm8k", "math", "svamp"]

# Mapping from base model directory name to HuggingFace tokenizer ID
HF_TOKENIZER_MAP = {
	"Qwen1.5-14B": "Qwen/Qwen1.5-14B",
	"Qwen1.5-7B": "Qwen/Qwen1.5-7B",
	"gemma-2-9b": "google/gemma-2-9b",
	"gemma-2-2b": "google/gemma-2-2b",
	"OLMo-2-1124-7B": "allenai/OLMo-2-1124-7B",
	"OLMo-2-0425-1B": "allenai/OLMo-2-0425-1B",
	"OLMo-2-1124-13B": "allenai/OLMo-2-1124-13B",
}

# Mapping from base model directory name to instruct model directory name
INSTRUCT_MODEL_MAP = {
	"Qwen1.5-14B": "Qwen1.5-14B-Chat",
	"Qwen1.5-7B": "Qwen1.5-7B-Chat",
	"gemma-2-9b": "gemma-2-9b-it",
	"gemma-2-2b": "gemma-2-2b-it",
	"OLMo-2-1124-7B": "OLMo-2-1124-7B-Instruct",
	"OLMo-2-0425-1B": "OLMo-2-0425-1B-Instruct",
}

# Tokenizer cache to avoid reloading
_tokenizer_cache = {}


def _get_tokenizer(model_name):
	"""Load and cache a HuggingFace tokenizer by model name."""
	if model_name not in _tokenizer_cache:
		hf_id = HF_TOKENIZER_MAP.get(model_name, model_name)
		_tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(hf_id)
	return _tokenizer_cache[model_name]


# Model filtering and name mapping
MODEL_FILTER = {
	"Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": "Qwen-1.5-14B",
	"gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": "gemma-2-9B",
	"OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": "OLMo-2-7B",
	# "Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": "Qwen-1.5-7B",
	# "gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": "gemma-2-2B",
	# "OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-1B",
}


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


PAPER_NAME = {
	"Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": "Qwen1.5-14B"
	+ f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;7B}}}}$",
	"gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": "gemma-2-9B"
	+ f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;2B}}}}$",
	"OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": "OLMo-2-7B"
	+ f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;1B}}}}$",
	"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": "Qwen1.5-7B"
	+ f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;14B}}}}$",
	"gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": "gemma-2-2B"
	+ f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;9B}}}}$",
	"OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-1B"
	+ f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;7B}}}}$",
}


def _apply_plot_style():
	# A clean, modern baseline without extra dependencies (e.g., seaborn).
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/JuliaMono-Regular.ttf"))
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Regular.ttf"))
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Bold.ttf"))

	plt.rcParams.update(
		{
			"text.usetex": False,
			"font.family": "monospace",
			"font.monospace": ["Noto Sans Mono", "JuliaMono", "DejaVu Sans Mono"],
			"mathtext.fontset": "dejavusans",
			"figure.dpi": 1000,
			"savefig.dpi": 1000,
			"axes.titlesize": 25,
			"axes.titleweight": "bold",
			"axes.labelsize": 25,
			"xtick.labelsize": 22,
			"axes.labelweight": "bold",
			"ytick.labelsize": 22,
			"legend.fontsize": 17,
			"hatch.linewidth": 0.5,
		}
	)


def scan_transfer_folder(folder_path, method_name):
	"""
	Scan a transfer folder and index all model configurations with their accuracy.

	Returns:
	    dict: Nested dict of {model_family: {dataset: [(accuracy, config_info)]}}
	"""
	results = defaultdict(lambda: defaultdict(list))

	if not folder_path.exists():
		print(f"Warning: {folder_path} does not exist")
		return results

	# Traverse the directory structure
	for num_train_dir in folder_path.iterdir():
		if not num_train_dir.is_dir() or not num_train_dir.name.startswith(
			"num_train_samples_"
		):
			continue

		num_train_samples = num_train_dir.name.replace("num_train_samples_", "")

		for model_family_dir in num_train_dir.iterdir():
			if not model_family_dir.is_dir():
				continue

			model_family = model_family_dir.name

			for rank_dir in model_family_dir.iterdir():
				if not rank_dir.is_dir() or not rank_dir.name.startswith("rank_"):
					continue

				rank = rank_dir.name.replace("rank_", "")

				for pca_steer_dir in rank_dir.iterdir():
					if not pca_steer_dir.is_dir():
						continue

					for dataset_dir in pca_steer_dir.iterdir():
						if not dataset_dir.is_dir():
							continue

						dataset_name = dataset_dir.name

						# Only process target datasets
						if dataset_name not in DATASETS:
							continue

						# Find metrics and jsonl files
						metrics_files = list(dataset_dir.glob("*_metrics.json"))
						jsonl_files = list(dataset_dir.glob("*.jsonl"))

						if metrics_files and jsonl_files:
							metrics_file = metrics_files[0]
							jsonl_file = jsonl_files[0]

							try:
								with open(metrics_file, "r") as f:
									metrics = json.load(f)
									accuracy = metrics.get("acc", 0.0)

								config_info = {
									"method": method_name,
									"model_family": model_family,
									"num_train_samples": num_train_samples,
									"rank": rank,
									"dataset": dataset_name,
									"accuracy": accuracy,
									"jsonl_path": str(jsonl_file),
									"metrics_path": str(metrics_file),
								}

								results[model_family][dataset_name].append(
									(accuracy, config_info)
								)

							except Exception as e:
								print(f"Error processing {metrics_file}: {e}")

	return results


def find_best_configs(avg_results, pca_results):
	"""
	Find the best configuration for each model family and dataset.
	Selects between avg and pca based on which has higher accuracy.

	Returns:
	    dict: {model_family: {dataset: config_info}}
	"""
	best_configs = defaultdict(dict)

	# Get all unique model families
	all_model_families = set(list(avg_results.keys()) + list(pca_results.keys()))

	for model_family in all_model_families:
		for dataset in DATASETS:
			best_avg = None
			best_pca = None

			# Find best in avg
			if model_family in avg_results and dataset in avg_results[model_family]:
				configs = avg_results[model_family][dataset]
				if configs:
					best_avg = max(configs, key=lambda x: x[0])[1]

			# Find best in pca
			if model_family in pca_results and dataset in pca_results[model_family]:
				configs = pca_results[model_family][dataset]
				if configs:
					best_pca = max(configs, key=lambda x: x[0])[1]

			# Select the method with higher accuracy
			if best_avg and best_pca:
				if best_avg["accuracy"] >= best_pca["accuracy"]:
					best_configs[model_family][dataset] = best_avg
				else:
					best_configs[model_family][dataset] = best_pca
			elif best_avg:
				best_configs[model_family][dataset] = best_avg
			elif best_pca:
				best_configs[model_family][dataset] = best_pca

	return best_configs


def calculate_gen_token_length(jsonl_path, model_name):
	"""
	Calculate average generation token count from a JSONL file using the model's tokenizer.

	Computes the mean token count over the *entire dataset* (i.e., all rows in the JSONL),
	using the first item in the `code` field when present.

	Returns:
		float: Average generation length in tokens, or None if no usable rows
	"""
	tokenizer = _get_tokenizer(model_name)
	token_counts = []

	try:
		with open(jsonl_path, "r") as f:
			for line in f:
				if not line.strip():
					continue

				try:
					data = json.loads(line)

					# Get the code field (which is a list)
					code = data.get("code", [])
					if code and len(code) > 0:
						# Use the first item in the code list
						code_text = code[0]
						token_count = len(
							tokenizer.encode(code_text, add_special_tokens=False)
						)
						token_counts.append(token_count)

				except json.JSONDecodeError as e:
					print(f"Error decoding JSON line in {jsonl_path}: {e}")
					continue

		if token_counts:
			return np.mean(token_counts)
		else:
			return None

	except Exception as e:
		print(f"Error reading {jsonl_path}: {e}")
		return None


def extract_base_model_name(model_family):
	"""
	Extract the base model name from a model family string.
	E.g., "gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b" -> "gemma-2-9b"
	"""
	# Split by the first occurrence of '_+_' or '_-_'
	parts = model_family.split("_+_")
	if len(parts) > 1:
		return parts[0]

	parts = model_family.split("_-_")
	if len(parts) > 1:
		return parts[0]

	return model_family


def process_baseline(best_configs):
	"""
	Process baseline evaluations for each model family and dataset.
	Extracts generation lengths for both direct and CoT prompts.

	Returns:
	    dict: {model_family: {dataset: {'direct': length, 'cot': length}}}
	"""
	baseline_results = defaultdict(lambda: defaultdict(dict))

	for model_family, datasets in best_configs.items():
		if model_family not in MODEL_FILTER:
			continue
		base_model = extract_base_model_name(model_family)
		if base_model not in HF_TOKENIZER_MAP:
			continue
		base_model_dir = BASE_EVAL_DIR / base_model

		if not base_model_dir.exists():
			print(f"Warning: Baseline directory not found for {base_model}")
			continue

		for dataset in datasets.keys():
			dataset_dir = base_model_dir / dataset

			if not dataset_dir.exists():
				print(f"Warning: Dataset directory not found: {dataset_dir}")
				continue

			# Find direct and CoT files
			direct_files = list(dataset_dir.glob("*_prompt_general-direct.jsonl"))
			cot_files = list(dataset_dir.glob("*_prompt_general-cot.jsonl"))

			# Process direct
			if direct_files:
				direct_length = calculate_gen_token_length(direct_files[0], base_model)
				if direct_length is not None:
					baseline_results[model_family][dataset]["direct"] = direct_length

			# Process CoT
			if cot_files:
				cot_length = calculate_gen_token_length(cot_files[0], base_model)
				if cot_length is not None:
					baseline_results[model_family][dataset]["cot"] = cot_length

			# Process Instruct CoT
			instruct_model = INSTRUCT_MODEL_MAP.get(base_model)
			if instruct_model:
				instruct_dir = BASE_EVAL_DIR / instruct_model / dataset
				if instruct_dir.exists():
					instruct_cot_files = list(
						instruct_dir.glob("*_prompt_general-cot.jsonl")
					)
					if instruct_cot_files:
						instruct_cot_length = calculate_gen_token_length(
							instruct_cot_files[0], base_model
						)
						if instruct_cot_length is not None:
							baseline_results[model_family][dataset]["instruct_cot"] = (
								instruct_cot_length
							)

	return baseline_results


def create_visualization(best_configs, baseline_results):
	"""
	Create a grouped bar plot comparing baseline and transfer methods.
	"""
	_apply_plot_style()

	# Organize data for plotting
	plot_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

	# Collect data
	for model_family in best_configs.keys():
		if model_family not in MODEL_FILTER:
			continue
		for dataset in DATASETS:
			# Get baseline data
			if (
				model_family in baseline_results
				and dataset in baseline_results[model_family]
			):
				base = baseline_results[model_family][dataset]
				plot_data[dataset][model_family]["base_direct"] = base.get("direct", 0)
				plot_data[dataset][model_family]["instruct_cot"] = base.get(
					"instruct_cot", 0
				)

			# Get transfer data
			if dataset in best_configs[model_family]:
				config = best_configs[model_family][dataset]
				transfer_length = calculate_gen_token_length(
					config["jsonl_path"], extract_base_model_name(model_family)
				)
				if transfer_length is not None:
					plot_data[dataset][model_family]["transfer"] = transfer_length

	# Filter to only include specified models and those with data
	valid_model_families = []
	for model_family in MODEL_FILTER.keys():
		# Check if this model has data for at least one dataset
		has_data = False
		for dataset in DATASETS:
			if model_family in plot_data[dataset]:
				data = plot_data[dataset][model_family]
			if (
				data.get("base_direct", 0) > 0
				or data.get("instruct_cot", 0) > 0
				or data.get("transfer", 0) > 0
			):
				has_data = True
				break
		if has_data:
			valid_model_families.append(model_family)

	# Sort for consistency
	valid_model_families = sorted(valid_model_families)

	if not valid_model_families:
		print("No valid data to plot!")
		return

	# Setup the plot with tighter spacing
	fig, ax = plt.subplots(figsize=(10, 8))

	# Bar configuration
	bar_width = 0.18
	methods = ["base_direct", "instruct_cot", "transfer"]
	method_labels = [
		"$\mathcal{T}_L$ (Base, Direct)",
		r"$\mathcal{T}^*_{\mathrm{PT}}$ (Instruct, CoT)",
		"$\mathcal{T}_U$ (Steered, Direct)",
	]
	# User-provided colors for each method per model family
	# Each model gets its own palette of 3 distinct colors (base_direct, instruct_cot, steered)
	# Format: {model_family: [base_direct_color, instruct_color, steered_color]}
	MODEL_METHOD_COLORS = {
		"Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": [  ## GREEN
			"#BEF2E8",
			"#5ED1B3",
			"#03916D",
		],
		"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": [  ## GREEN
			"#BEF2E8",
			"#5ED1B3",
			"#03916D",
		],
		"gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": [  ## BLUE
			"#C2DFF2",
			"#57ADD9",
			"#0072B2",
		],
		"gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": [  ## BLUE
			"#C2DFF2",
			"#57ADD9",
			"#0072B2",
		],
		"OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": [  ## RED
			"#FBF3E9",
			"#EECA9B",
			"#D98C25",
		],
		"OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": [  ## RED
			"#FBF3E9",
			"#EECA9B",
			"#D98C25",
		],
	}

	# X positions for datasets with tighter spacing
	dataset_positions = np.arange(len(DATASETS))
	spacing_between_datasets = (
		len(valid_model_families) * len(methods) * bar_width + 0.45
	)

	x_positions = dataset_positions * spacing_between_datasets

	# Plot bars
	all_bar_x = []
	for dataset_idx, dataset in enumerate(DATASETS):
		base_x = x_positions[dataset_idx]

		for model_idx, model_family in enumerate(valid_model_families):
			model_data = plot_data[dataset][model_family]

			for method_idx, method in enumerate(methods):
				value = model_data.get(method, 0)

				x_pos = (
					base_x
					+ model_idx * len(methods) * bar_width
					+ method_idx * bar_width
				)

				# Create labels only for the first dataset to avoid duplicates
				label = None
				if dataset_idx == 0:
					if method_idx == 0:
						# Label with clean model name (only once per model)
						label = MODEL_FILTER[model_family]

				color = MODEL_METHOD_COLORS.get(model_family, ["#888888"] * 3)[
					method_idx
				]

				ax.bar(
					x_pos,
					value,
					bar_width,
					label=label,
					color=color,
					edgecolor="#000000",
					linewidth=0.5,
					alpha=1,
				)
				all_bar_x.append(x_pos)

	# Customize the plot
	# ax.set_xlabel("Dataset")
	ax.set_ylabel("Avg. Generation Length (Tokens)")

	# Set x-tick positions to center of each dataset group
	dataset_centers = []
	for dataset_idx in range(len(DATASETS)):
		base_x = x_positions[dataset_idx]
		total_width = len(valid_model_families) * len(methods) * bar_width
		center = base_x + total_width / 2 - bar_width / 2
		dataset_centers.append(center)

	ax.set_xticks(dataset_centers)
	ax.set_xticklabels([d.upper() for d in DATASETS], fontweight="bold")

	# Reduce whitespace before the first bar by tightening x-limits to the bars.
	if all_bar_x:
		min_x = min(all_bar_x) - bar_width * 0.65
		max_x = max(all_bar_x) + bar_width * 1.65
		ax.set_xlim(min_x, max_x)
	ax.margins(x=0.0)
	ax.set_ylim(0, 450)

	from matplotlib.patches import Patch

	# Legend for model families (use middle color from each model palette)
	model_legend_elements = []
	for model in valid_model_families:
		display = PAPER_NAME.get(model, model)
		model_legend_elements.append(
			Patch(
				facecolor=MODEL_METHOD_COLORS.get(model, ["#888888"] * 3)[1],
				edgecolor="#000000",
				label=display,
			)
		)

	# Gray shades for method legend (light to darker)
	method_gray_colors = ["#DDDDDD", "#AAAAAA", "#777777"]
	# Legend for methods (use gray shades)
	method_legend_elements = [
		Patch(
			facecolor=method_gray_colors[i],
			edgecolor="#000000",
			label=method_labels[i],
		)
		for i in range(len(methods))
	]

	# Two compact legends (models = colors, methods = hatches), placed outside.
	leg1 = ax.legend(
		handles=model_legend_elements + method_legend_elements,
		loc="upper right",
		bbox_to_anchor=(1.0, 1.0),
		borderaxespad=0.0,
		framealpha=0.9,
		ncol=2,
	)
	ax.add_artist(leg1)

	# Add grid for better readability
	ax.grid(which="major", axis="both")
	ax.set_axisbelow(True)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.08)
	# Save the plot
	output_path = Path("figures/gen_length_bars_s2l.pdf")
	plt.savefig(output_path, dpi=300)
	print(f"\nPlot saved to: {output_path}")


def main():
	print("=" * 80)
	print("GENERATION LENGTH ANALYSIS")
	print("=" * 80)

	# Step 1: Scan transfer folders
	print("\n[1/5] Scanning transfer folders...")
	avg_results = scan_transfer_folder(AVG_DIR, "avg")
	pca_results = scan_transfer_folder(PCA_DIR, "pca")

	print(f"  Found {len(avg_results)} model families in AVG folder")
	print(f"  Found {len(pca_results)} model families in PCA folder")

	# Step 2: Find best configurations
	print("\n[2/5] Finding best configurations per model family and dataset...")
	best_configs = find_best_configs(avg_results, pca_results)

	print(f"  Best configs identified for {len(best_configs)} model families")
	for model_family, datasets in best_configs.items():
		print(f"    {model_family}: {list(datasets.keys())}")

	# Step 3: Calculate transfer generation lengths (done in visualization step)
	print("\n[3/5] Will calculate transfer generation lengths during visualization...")

	# Step 4: Process baseline results
	print("\n[4/5] Processing baseline evaluations...")
	baseline_results = process_baseline(best_configs)

	print(baseline_results)

	print(f"  Processed baseline results for {len(baseline_results)} model families")

	# Step 5: Create visualization
	print("\n[5/5] Creating visualization...")
	create_visualization(best_configs, baseline_results)

	print("\n" + "=" * 80)
	print("ANALYSIS COMPLETE")
	print("=" * 80)


if __name__ == "__main__":
	main()
