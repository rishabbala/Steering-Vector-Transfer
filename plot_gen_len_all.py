#!/usr/bin/env python3
"""
Generation Token Length Analysis for Transfer Learning Methods
Compares base model (direct, no CoT) with unlocked model (best transfer config)
across multiple datasets and model families using a scatter plot.
"""

import json
from collections import defaultdict
from pathlib import Path
import os
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
import numpy as np
from matplotlib.lines import Line2D
from transformers import AutoTokenizer

# Paths to data directories
BASE_DIR = Path("/projects/llms-lab/transfer_compare")
AVG_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_avg"
PCA_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_pca"
BASE_EVAL_DIR = BASE_DIR / "base_evals_test"

# Target datasets
DATASETS = ["math", "gsm8k", "svamp"]

# Model filtering and name mapping
MODEL_FILTER = {
	"Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": "Qwen1.5-14B",
	"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": "Qwen1.5-7B",
	"OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-0425-1B",
	"OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": "OLMo-2-1124-7B",
	"OLMo-2-1124-13B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-1124-13B",
	"gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": "gemma-2-9b",
	"gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": "gemma-2-2b",
}

PAPER_NAME = {
	"Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": "Qwen1.5-14B",
	"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": "Qwen1.5-7B",
	"gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": "Gemma-2-9B",
	"OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": "OLMo-2-7B",
	"gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": "Gemma-2-2B",
	"OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-1B",
	"OLMo-2-1124-13B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-13B",
}

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

# Different marker shapes per model
MARKERS = ["o", "s", "^", "D", "X", "*", "v"]

# Tokenizer cache to avoid reloading
_tokenizer_cache = {}


def _get_tokenizer(model_name):
	"""Load and cache a HuggingFace tokenizer by model name."""
	if model_name not in _tokenizer_cache:
		hf_id = HF_TOKENIZER_MAP.get(model_name, model_name)
		print("*" * 100)
		print(model_name)
		print(hf_id)
		print("*" * 100)
		print(f"  Loading tokenizer for {model_name} -> {hf_id}")
		_tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(hf_id)
	return _tokenizer_cache[model_name]


def _apply_plot_style():
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
			"axes.titlesize": 28,
			"axes.titleweight": "bold",
			"axes.labelsize": 24,
			"xtick.labelsize": 22,
			"axes.labelweight": "bold",
			"ytick.labelsize": 22,
			"legend.fontsize": 16,
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
						if dataset_name not in DATASETS:
							continue

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

	all_model_families = set(list(avg_results.keys()) + list(pca_results.keys()))
	all_model_families = [m for m in all_model_families if m in MODEL_FILTER]

	for model_family in all_model_families:
		for dataset in DATASETS:
			best_avg = None
			best_pca = None

			if model_family in avg_results and dataset in avg_results[model_family]:
				configs = avg_results[model_family][dataset]
				if configs:
					best_avg = max(configs, key=lambda x: x[0])[1]

			if model_family in pca_results and dataset in pca_results[model_family]:
				configs = pca_results[model_family][dataset]
				if configs:
					best_pca = max(configs, key=lambda x: x[0])[1]

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

	Computes the mean token count over the entire dataset using the first item
	in the `code` field when present.

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
					code = data.get("code", [])
					if code and len(code) > 0:
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
	Extracts generation token lengths for the direct (no CoT) prompt.

	Returns:
	    dict: {model_family: {dataset: {'direct': token_length}}}
	"""
	baseline_results = defaultdict(lambda: defaultdict(dict))

	for model_family, datasets in best_configs.items():
		if model_family not in MODEL_FILTER:
			continue
		base_model = extract_base_model_name(model_family)
		base_model_dir = BASE_EVAL_DIR / base_model

		if not base_model_dir.exists():
			print(f"Warning: Baseline directory not found for {base_model}")
			continue

		for dataset in datasets.keys():
			dataset_dir = base_model_dir / dataset

			if not dataset_dir.exists():
				print(f"Warning: Dataset directory not found: {dataset_dir}")
				continue

			direct_files = list(dataset_dir.glob("*_prompt_general-direct.jsonl"))
			instruct_files = list(dataset_dir.glob("*_prompt_general-instruct.jsonl"))

			if direct_files:
				direct_tokens = calculate_gen_token_length(direct_files[0], base_model)
				if direct_tokens is not None:
					baseline_results[model_family][dataset]["direct"] = direct_tokens

			if instruct_files:
				instruct_tokens = calculate_gen_token_length(
					instruct_files[0], base_model
				)
				if instruct_tokens is not None:
					baseline_results[model_family][dataset]["instruct"] = (
						instruct_tokens
					)

	return baseline_results


def print_results_table(best_configs, baseline_results):
	"""
	Print a formatted table of average token counts for each model-dataset pair.
	Columns: Model | Dataset | Direct | Instruct | Transfer (Unlocked)
	Also prints per-model averages across datasets.
	"""
	# Collect transfer token counts
	transfer_results = defaultdict(dict)
	for model_family in best_configs.keys():
		if model_family not in MODEL_FILTER:
			continue
		base_model = extract_base_model_name(model_family)
		if base_model not in HF_TOKENIZER_MAP:
			continue
		for dataset in DATASETS:
			if dataset in best_configs[model_family]:
				config = best_configs[model_family][dataset]
				tokens = calculate_gen_token_length(config["jsonl_path"], base_model)
				if tokens is not None:
					transfer_results[model_family][dataset] = tokens

	# Print header
	model_col = 40
	dataset_col = 12
	num_col = 12
	sep = "-" * (model_col + dataset_col + num_col * 3)

	print(f"\n{sep}")
	print(
		f"{'Model':<{model_col}}{'Dataset':<{dataset_col}}"
		f"{'Direct':<{num_col}}{'Instruct':<{num_col}}{'Transfer':<{num_col}}"
	)
	print(sep)

	# Per-model averages
	model_totals = defaultdict(lambda: {"direct": [], "instruct": [], "transfer": []})

	for model_family in sorted(MODEL_FILTER.keys()):
		display_name = MODEL_FILTER[model_family]
		for dataset in DATASETS:
			direct = (
				baseline_results.get(model_family, {})
				.get(dataset, {})
				.get("direct", None)
			)
			instruct = (
				baseline_results.get(model_family, {})
				.get(dataset, {})
				.get("instruct", None)
			)
			transfer = transfer_results.get(model_family, {}).get(dataset, None)

			direct_str = f"{direct:.1f}" if direct is not None else "-"
			instruct_str = f"{instruct:.1f}" if instruct is not None else "-"
			transfer_str = f"{transfer:.1f}" if transfer is not None else "-"

			if direct is not None:
				model_totals[model_family]["direct"].append(direct)
			if instruct is not None:
				model_totals[model_family]["instruct"].append(instruct)
			if transfer is not None:
				model_totals[model_family]["transfer"].append(transfer)

			print(
				f"{display_name:<{model_col}}{dataset:<{dataset_col}}"
				f"{direct_str:<{num_col}}{instruct_str:<{num_col}}{transfer_str:<{num_col}}"
			)

		# Model average row
		totals = model_totals[model_family]
		avg_direct = np.mean(totals["direct"]) if totals["direct"] else None
		avg_instruct = np.mean(totals["instruct"]) if totals["instruct"] else None
		avg_transfer = np.mean(totals["transfer"]) if totals["transfer"] else None

		avg_direct_str = f"{avg_direct:.1f}" if avg_direct is not None else "-"
		avg_instruct_str = f"{avg_instruct:.1f}" if avg_instruct is not None else "-"
		avg_transfer_str = f"{avg_transfer:.1f}" if avg_transfer is not None else "-"

		print(
			f"{display_name + ' AVG':<{model_col}}{'-':<{dataset_col}}"
			f"{avg_direct_str:<{num_col}}{avg_instruct_str:<{num_col}}{avg_transfer_str:<{num_col}}"
		)
		print(sep)

	print()


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


def create_visualization(best_configs, baseline_results):
	"""
	Create a scatter plot comparing base model (direct) vs unlocked model (transfer)
	token counts across datasets.

	Base models: red markers with different shapes per model.
	Unlocked models: blue markers with matching shapes per model.
	"""
	_apply_plot_style()

	# Collect token count data
	plot_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

	for model_family in best_configs.keys():
		if model_family not in MODEL_FILTER:
			continue
		for dataset in DATASETS:
			if (
				model_family in baseline_results
				and dataset in baseline_results[model_family]
			):
				base = baseline_results[model_family][dataset]
				plot_data[dataset][model_family]["base_direct"] = base.get("direct", 0)

			if dataset in best_configs[model_family]:
				config = best_configs[model_family][dataset]
				base_model = extract_base_model_name(model_family)
				transfer_tokens = calculate_gen_token_length(
					config["jsonl_path"], base_model
				)
				if transfer_tokens is not None:
					plot_data[dataset][model_family]["transfer"] = transfer_tokens

	# Filter to only include specified models with data
	valid_model_families = []
	for model_family in MODEL_FILTER.keys():
		has_data = False
		for dataset in DATASETS:
			if model_family in plot_data[dataset]:
				data = plot_data[dataset][model_family]
				if data.get("base_direct", 0) > 0 or data.get("transfer", 0) > 0:
					has_data = True
					break
		if has_data:
			valid_model_families.append(model_family)

	valid_model_families = sorted(valid_model_families)

	if not valid_model_families:
		print("No valid data to plot!")
		return

	# Assign marker shapes per model
	model_markers = {
		model: MARKERS[i % len(MARKERS)] for i, model in enumerate(valid_model_families)
	}

	fig, ax = plt.subplots(figsize=(8, 6))

	dataset_x = {d: i for i, d in enumerate(DATASETS)}

	for model_family in valid_model_families:
		marker = model_markers[model_family]
		for dataset in DATASETS:
			if model_family not in plot_data[dataset]:
				continue
			data = plot_data[dataset][model_family]
			x = dataset_x[dataset]

			base_val = data.get("base_direct", None)
			transfer_val = data.get("transfer", None)

			if base_val is not None and base_val > 0:
				ax.scatter(
					x,
					base_val,
					c="#D32F2F",
					marker=marker,
					s=250,
					zorder=3,
					edgecolors="#333333",
					linewidths=0.5,
					alpha=0.6,
				)

			if transfer_val is not None and transfer_val > 0:
				ax.scatter(
					x,
					transfer_val,
					c="#1976D2",
					marker=marker,
					s=250,
					zorder=3,
					edgecolors="#333333",
					linewidths=0.5,
					alpha=0.6,
				)

	# Build legend
	legend_handles = []

	# Model shapes (using a neutral color for the shape legend)
	for model in valid_model_families:
		display = PAPER_NAME.get(model, model)
		legend_handles.append(
			Line2D(
				[0],
				[0],
				marker=model_markers[model],
				color="w",
				markerfacecolor="#888888",
				markeredgecolor="#333333",
				markeredgewidth=0.5,
				markersize=14,
				label=display,
			)
		)

	# Color legend for base vs unlocked
	# legend_handles.append(
	# 	Line2D(
	# 		[0],
	# 		[0],
	# 		marker="o",
	# 		color="w",
	# 		markerfacecolor="#D32F2F",
	# 		markeredgecolor="#333333",
	# 		markeredgewidth=0.5,
	# 		markersize=16,
	# 		label=r"$\mathcal{T}_{\mathrm{L}}$",
	# 	)
	# )
	# legend_handles.append(
	# 	Line2D(
	# 		[0],
	# 		[0],
	# 		marker="o",
	# 		color="w",
	# 		markerfacecolor="#1976D2",
	# 		markeredgecolor="#333333",
	# 		markeredgewidth=0.5,
	# 		markersize=16,
	# 		label=r"$\mathcal{T}_{\mathrm{U}}$",
	# 	)
	# )

	legend_handles.append(
		Line2D(
			[0],
			[0],
			marker="o",
			color="w",
			markerfacecolor="#D32F2F",
			markeredgecolor="#333333",
			markeredgewidth=0.5,
			markersize=14,
			label=r"Base model",
		)
	)
	legend_handles.append(
		Line2D(
			[0],
			[0],
			marker="o",
			color="w",
			markerfacecolor="#1976D2",
			markeredgecolor="#333333",
			markeredgewidth=0.5,
			markersize=14,
			label=f"Base model + {to_small_caps('unlock')}",
		)
	)

	ax.legend(
		handles=legend_handles,
		loc="upper right",
		# borderaxespad=0.5,
		framealpha=0.9,
		ncol=2,
		labelspacing=0.2,  # vertical gap between rows (default 0.5)
		columnspacing=0.2,  # gap between columns (default 2.0)
	)

	# ax.set_xlabel("Dataset")
	ax.set_ylabel("Avg. Generation Length\n(Tokens)")
	ax.set_xticks(range(len(DATASETS)))
	ax.set_xticklabels([d.upper() for d in DATASETS], fontweight="bold")
	ax.grid(which="major", axis="both")
	ax.set_axisbelow(True)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.set_ylim(0, 350)
	plt.subplots_adjust(left=0.19, right=0.97, top=0.94, bottom=0.12)

	output_path = Path("figures/gen_len_all.png")
	plt.savefig(output_path)
	print(f"\nPlot saved to: {output_path}")


def main():
	print("=" * 80)
	print("GENERATION TOKEN LENGTH ANALYSIS")
	print("=" * 80)

	print("\n[1/4] Scanning transfer folders...")
	avg_results = scan_transfer_folder(AVG_DIR, "avg")
	pca_results = scan_transfer_folder(PCA_DIR, "pca")

	print(f"  Found {len(avg_results)} model families in AVG folder")
	print(f"  Found {len(pca_results)} model families in PCA folder")

	print("\n[2/4] Finding best configurations per model family and dataset...")
	best_configs = find_best_configs(avg_results, pca_results)

	print(f"  Best configs identified for {len(best_configs)} model families")
	for model_family, datasets in best_configs.items():
		print(f"    {model_family}: {list(datasets.keys())}")

	print("\n[3/4] Processing baseline evaluations (token counting)...")
	baseline_results = process_baseline(best_configs)

	print(f"  Processed baseline results for {len(baseline_results)} model families")

	print("\n[4/5] Printing results table...")
	print_results_table(best_configs, baseline_results)

	print("\n[5/5] Creating visualization...")
	create_visualization(best_configs, baseline_results)

	print("\n" + "=" * 80)
	print("ANALYSIS COMPLETE")
	print("=" * 80)


if __name__ == "__main__":
	main()
