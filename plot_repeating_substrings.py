import json
import os
import string
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

PAPER_DATASET_NAME = {
	"agieval_math": "Agieval Math",
	"minerva_math": "Minerva Math",
	"olympiadbench": "Olympiad Bench",
	"deepmind_math": "Deepmind Math",
}

# ROOT_DIR = "/projects/llms-lab/transfer_compare"
IN_PTH = [
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_avg/num_train_samples_64/Ministral-3-8B-Base-2512_+_Ministral-3-3B-Instruct-2512-BF16_-_Ministral-3-3B-Base-2512/rank_64/pca_math_steering_math/agieval_math/mistralai_Ministral-3-8B-Base-2512_alpha_0.05_test_num_samples_-1_max_gen_tokens_4096.jsonl",
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_ID_pca/num_train_samples_32/Ministral-3-8B-Base-2512_+_Ministral-3-3B-Instruct-2512-BF16_-_Ministral-3-3B-Base-2512/rank_16/pca_deepmind_math_steering_deepmind_math/deepmind_math/mistralai_Ministral-3-8B-Base-2512_alpha_0.05_test_num_samples_-1_max_gen_tokens_4096.jsonl",
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_ID_avg/num_train_samples_32/Ministral-3-8B-Base-2512_+_Ministral-3-3B-Instruct-2512-BF16_-_Ministral-3-3B-Base-2512/rank_1/pca_minerva_math_steering_minerva_math/minerva_math/mistralai_Ministral-3-8B-Base-2512_alpha_0.1_test_num_samples_-1_max_gen_tokens_4096.jsonl",
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca/num_train_samples_64/Ministral-3-8B-Base-2512_+_Ministral-3-3B-Instruct-2512-BF16_-_Ministral-3-3B-Base-2512/rank_4/pca_math_steering_math/olympiadbench/mistralai_Ministral-3-8B-Base-2512_alpha_0.05_test_num_samples_-1_max_gen_tokens_4096.jsonl",
]


if not os.path.exists("./figures"):
	os.makedirs("./figures")

base_colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
light_colors = ["#99C7E0", "#F5D999", "#99D8C7", "#EEBF99", "#EBC9DC"]


def get_base_path(steered_path):
	if "num_train_samples_" in steered_path:
		parts = steered_path.split("/")
		model_with_base = parts[6].split("_+_")[0]
		model_with_base2 = parts[10].split("_alpha")[0]
		print(model_with_base)
		dataset = steered_path.split("steering_")[-1].split("/")[1]
		base_path = f"/projects/llms-lab/transfer_compare/base_evals_test/{model_with_base}/{dataset}/{model_with_base2}_test_num_samples_-1_max_gen_tokens_4096_prompt_general-cot.jsonl"
		return base_path
	return None


def get_average_length(file_path, correct_only=True):
	data = []
	with open(file_path, "r") as file:
		for line in file:
			data.append(json.loads(line))

	total_length = 0
	count = 0

	for d in data:
		is_correct = d["score"][0] is True
		if (correct_only and is_correct) or (not correct_only and not is_correct):
			code_text = d["code"][0]
			total_length += len(code_text)
			count += 1

	avg_length = total_length / count if count > 0 else 0
	return avg_length


def has_repeating_substring(text, length):
	if len(text) < 2 * length:
		return False

	seen = set()
	for i in range(len(text) - length + 1):
		substring = text[i : i + length]
		if substring in seen:
			return True
		seen.add(substring)
	return False


def get_longest_repeating_substring_length(text):
	max_length = 0
	for length in range(1, len(text) // 2 + 1):
		seen = set()
		found = False
		for i in range(len(text) - length + 1):
			substring = text[i : i + length]
			if substring in seen:
				max_length = length
				found = True
				break
			seen.add(substring)
		if not found:
			break
	return max_length


def get_repetition_distribution(file_path, correct_only=False):
	data = []
	with open(file_path, "r") as file:
		for line in file:
			data.append(json.loads(line))

	substring_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
	counts = {l: 0 for l in substring_lengths}
	most_frequent = {l: [] for l in substring_lengths}

	for d in data:
		is_correct = d["score"][0] is True
		if (correct_only and is_correct) or (not correct_only and not is_correct):
			code_text = d["code"][0]
			for length in substring_lengths:
				seen = {}
				for i in range(len(code_text) - length + 1):
					substring = code_text[i : i + length]
					if substring in seen:
						seen[substring] += 1
					else:
						seen[substring] = 1

				repeated = {s: c for s, c in seen.items() if c > 1}
				if repeated:
					counts[length] += 1
					max_repeat = max(repeated.items(), key=lambda x: x[1])
					most_frequent[length].append((max_repeat[0], max_repeat[1]))

	return counts, most_frequent


def get_lcs_distribution(file_path, correct_only=False):
	data = []
	with open(file_path, "r") as file:
		for line in file:
			data.append(json.loads(line))

	import math

	bins = {}
	for x in range(2, 13):
		bins[x] = 0

	for d in data:
		is_correct = d["score"][0] is True
		if (correct_only and is_correct) or (not correct_only and not is_correct):
			code_text = d["code"][0]
			lcs_length = get_longest_repeating_substring_length(code_text)

			if lcs_length > 0:
				x = int(math.log2(lcs_length))
				if x in bins:
					bins[x] += 1
				elif x > 12:
					bins[12] += 1

	return bins


def plot_gen_length_comparison(ax, correct_only=True):
	import matplotlib.colors as mcolors

	datasets = []
	base_lengths = []
	steered_lengths = []

	for steered_path in IN_PTH:
		dataset_name = steered_path.split("steering_")[-1].split("/")[1]
		datasets.append(dataset_name)

		steered_len = get_average_length(steered_path, correct_only=correct_only)
		steered_lengths.append(steered_len)

		base_path = get_base_path(steered_path)
		if base_path and os.path.exists(base_path):
			base_len = get_average_length(base_path, correct_only=correct_only)
			base_lengths.append(base_len)

			answer_type = "Correct" if correct_only else "Incorrect"
			print(f"{dataset_name} ({answer_type}) - Locked: {base_len:.2f} chars")
			print(f"{dataset_name} ({answer_type}) - Unlocked: {steered_len:.2f} chars")
		else:
			print(f"Base path not found: {base_path}")
			base_lengths.append(0)

	x = range(len(datasets))
	width = 0.35

	for i, dataset in enumerate(datasets):
		base_color = base_colors[i % len(base_colors)]
		light_color = light_colors[i % len(light_colors)]

		rgb = mcolors.to_rgb(base_color)
		light_rgb = mcolors.to_rgb(light_color)
		lighter_rgb = tuple(c for c in light_rgb)
		darker_rgb = tuple(c for c in rgb)

		locked_label = f"{PAPER_DATASET_NAME.get(dataset, dataset)}" if i == 0 else ""
		unlocked_label = ""

		ax.bar(
			i - width / 2,
			base_lengths[i],
			width,
			color=lighter_rgb,
			# hatch="/",
			edgecolor="black",
			linewidth=0.5,
			label=locked_label,
		)
		ax.bar(
			i + width / 2,
			steered_lengths[i],
			width,
			color=darker_rgb,
			# hatch="O",
			edgecolor="black",
			linewidth=0.5,
			label=unlocked_label,
		)

	from matplotlib.patches import Patch

	method_handles = [
		Patch(
			facecolor="#B0B0B0",
			edgecolor="black",
			# hatch="/",
			label=r"Target Locked $\mathcal{{T}}_\mathrm{{L}}$",
		),
		Patch(
			facecolor="#656565",
			edgecolor="black",
			# hatch="O",
			label=r"Target Unlocked $\mathcal{{T}}_\mathrm{{U}}$",
		),
	]

	dataset_handles = [
		Patch(
			facecolor=base_colors[i % len(base_colors)],
			edgecolor="black",
			label=PAPER_DATASET_NAME.get(d, d),
		)
		for i, d in enumerate(datasets)
	]

	all_handles_bar = dataset_handles + method_handles

	answer_type = "Correct" if correct_only else "Incorrect"
	# ax.set_xlabel("Dataset")
	ax.set_ylabel("Avg. Length (chars)")
	ax.set_title(
		f"Length-to-Answer",
	)
	ax.set_xticks(x)
	ax.set_xticklabels(
		[("\n".join(PAPER_DATASET_NAME.get(d, d).split(" "))) for d in datasets],
	)
	all_handles_bar_formatted = dataset_handles + method_handles

	ax.legend(
		handles=all_handles_bar_formatted,
		loc="upper left",
		ncol=1,
		# handlelength=2.5,
		# handleheight=1.5,
	)
	ax.grid(axis="y", alpha=0.3)
	ax.tick_params(axis="y")

	return dataset_handles


def plot_incorrect_repetition_distribution(ax):
	import matplotlib.colors as mcolors

	substring_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

	for idx, steered_path in enumerate(IN_PTH):
		dataset_name = steered_path.split("steering_")[-1].split("/")[1]
		base_color = base_colors[idx % len(base_colors)]
		light_color = light_colors[idx % len(light_colors)]

		rgb = mcolors.to_rgb(base_color)
		light_rgb = mcolors.to_rgb(light_color)
		lighter_rgb = tuple(c for c in light_rgb)
		darker_rgb = tuple(c for c in rgb)

		steered_counts, steered_freq = get_repetition_distribution(
			steered_path, correct_only=False
		)
		steered_values = [steered_counts[l] for l in substring_lengths]

		base_path = get_base_path(steered_path)
		if base_path and os.path.exists(base_path):
			base_counts, base_freq = get_repetition_distribution(
				base_path, correct_only=False
			)
			base_values = [base_counts[l] for l in substring_lengths]

			locked_label = (
				f"{PAPER_DATASET_NAME.get(dataset_name, dataset_name)}"
				if idx == 0
				else ""
			)
			unlocked_label = ""

			ax.plot(
				substring_lengths,
				base_values,
				marker="o",
				linewidth=2,
				linestyle="-",
				color=lighter_rgb,
				label=locked_label,
				markersize=6,
			)
			ax.plot(
				substring_lengths,
				steered_values,
				marker="s",
				linestyle="--",
				linewidth=2,
				color=darker_rgb,
				label=unlocked_label,
				markersize=6,
			)

			print(f"\n{dataset_name} - Locked repetition counts: {base_values}")
			print(f"{dataset_name} - Unlocked repetition counts: {steered_values}")

			print(f"\n{dataset_name} - Most repeated substrings within text (Locked):")
			for length in substring_lengths:
				if base_freq[length]:
					max_repeat_item = max(base_freq[length], key=lambda x: x[1])
					substring_repr = (
						repr(max_repeat_item[0][:50])
						if len(max_repeat_item[0]) > 50
						else repr(max_repeat_item[0])
					)
					print(
						f"  Length {length}: {substring_repr} (repeated {max_repeat_item[1]} times within a single text)"
					)

			print(
				f"\n{dataset_name} - Most repeated substrings within text (Unlocked):"
			)
			for length in substring_lengths:
				if steered_freq[length]:
					max_repeat_item = max(steered_freq[length], key=lambda x: x[1])
					substring_repr = (
						repr(max_repeat_item[0][:50])
						if len(max_repeat_item[0]) > 50
						else repr(max_repeat_item[0])
					)
					print(
						f"  Length {length}: {substring_repr} (repeated {max_repeat_item[1]} times within a single text)"
					)
		else:
			print(f"Base path not found: {base_path}")

	from matplotlib.patches import Patch

	method_handles_line = [
		plt.Line2D(
			[0],
			[0],
			color="#B0B0B0",
			marker="o",
			markersize=6,
			linestyle="-",
			linewidth=2,
			label=r"Target Locked $\mathcal{{T}}_\mathrm{{L}}$",
		),
		plt.Line2D(
			[0],
			[0],
			color="#656565",
			marker="s",
			markersize=6,
			linestyle="--",
			linewidth=2,
			label=r"Target Unlocked $\mathcal{{T}}_\mathrm{{U}}$",
		),
	]

	dataset_handles_line = [
		plt.Line2D(
			[0],
			[0],
			color=base_colors[i % len(base_colors)],
			marker="o",
			markersize=6,
			linewidth=2,
			label=PAPER_DATASET_NAME.get(d, d),
		)
		for i, d in enumerate(
			[
				steered_path.split("steering_")[-1].split("/")[1]
				for steered_path in IN_PTH
			]
		)
	]

	all_handles_line_formatted = dataset_handles_line + method_handles_line

	ax.set_xlabel("Substring Length (chars)")
	ax.set_ylabel("# Incorrect Examples")
	ax.set_title("Incorrect Examples w/ Repetitions")
	ax.set_xscale("log", base=2)
	ax.set_xticks(substring_lengths)
	ax.set_xticklabels([str(l) for l in substring_lengths], rotation=45)
	ax.legend(
		handles=all_handles_line_formatted,
		loc="upper right",
		ncol=1,
		# handlelength=2.5,
		# handleheight=1.5,
	)
	ax.grid(True, alpha=0.3)
	ax.tick_params(axis="y")


def plot_lcs_distribution(ax):
	import matplotlib.colors as mcolors

	for idx, steered_path in enumerate(IN_PTH):
		dataset_name = steered_path.split("steering_")[-1].split("/")[1]
		base_color = base_colors[idx % len(base_colors)]
		light_color = light_colors[idx % len(light_colors)]

		rgb = mcolors.to_rgb(base_color)
		light_rgb = mcolors.to_rgb(light_color)
		lighter_rgb = tuple(c for c in light_rgb)
		darker_rgb = tuple(c for c in rgb)
		steered_bins = get_lcs_distribution(steered_path, correct_only=False)

		base_path = get_base_path(steered_path)
		if base_path and os.path.exists(base_path):
			base_bins = get_lcs_distribution(base_path, correct_only=False)

			x_values = sorted(steered_bins.keys())
			base_values = [base_bins[x] for x in x_values]
			steered_values = [steered_bins[x] for x in x_values]

			locked_label = (
				f"{PAPER_DATASET_NAME.get(dataset_name, dataset_name)}"
				if idx == 0
				else ""
			)
			unlocked_label = ""

			ax.plot(
				x_values,
				base_values,
				marker="o",
				linewidth=2,
				linestyle="-",
				color=lighter_rgb,
				label=locked_label,
				markersize=6,
			)
			ax.plot(
				x_values,
				steered_values,
				marker="s",
				linewidth=2,
				linestyle="--",
				color=darker_rgb,
				label=unlocked_label,
				markersize=6,
			)

			print(f"\n{dataset_name} - LCS distribution (Locked): {base_values}")
			print(f"{dataset_name} - LCS distribution (Unlocked): {steered_values}")
		else:
			print(f"Base path not found: {base_path}")

	method_handles_lcs = [
		plt.Line2D(
			[0],
			[0],
			color="#B0B0B0",
			marker="o",
			markersize=6,
			linestyle="-",
			linewidth=2,
			label=r"Target Locked $\mathcal{{T}}_\mathrm{{L}}$",
		),
		plt.Line2D(
			[0],
			[0],
			color="#656565",
			marker="s",
			markersize=6,
			linestyle="--",
			linewidth=2,
			label=r"Target Unlocked $\mathcal{{T}}_\mathrm{{U}}$",
		),
	]

	dataset_handles_lcs = [
		plt.Line2D(
			[0],
			[0],
			color=base_colors[i % len(base_colors)],
			marker="o",
			markersize=6,
			linestyle="-",
			linewidth=2,
			label=PAPER_DATASET_NAME.get(d, d),
		)
		for i, d in enumerate(
			[
				steered_path.split("steering_")[-1].split("/")[1]
				for steered_path in IN_PTH
			]
		)
	]

	all_handles_lcs_formatted = dataset_handles_lcs + method_handles_lcs

	ax.set_xlabel(
		r"Lower bound of bin ",
	)
	ax.set_ylabel(
		r"# Repeating Substrings",
	)
	ax.set_title(
		"Longest Repeating Substring",
	)
	ax.set_xticks(sorted(steered_bins.keys()))
	ax.legend(
		handles=all_handles_lcs_formatted,
		loc="upper right",
		ncol=1,
		# handlelength=2.5,
		# handleheight=1.5,
	)
	ax.grid(True, alpha=0.3)
	ax.tick_params(axis="y")


def main():
	from matplotlib.patches import Patch

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
			"axes.titlesize": 38,
			"axes.titleweight": "bold",
			"axes.labelsize": 35,
			"xtick.labelsize": 30,
			"axes.labelweight": "bold",
			"ytick.labelsize": 30,
			"legend.fontsize": 28,
			"hatch.linewidth": 0.5,
		}
	)

	MODEL_NAME = IN_PTH[0].split("num_train_samples")[1].split("/")[1]

	OUT_PTH_COMBINED = (
		f"./figures/substring_analysis/{MODEL_NAME}_repetition_analysis.pdf"
	)
	fig, axes = plt.subplots(1, 3, figsize=(40, 8))

	print("Generating plot for correct answers...")
	dataset_handles = plot_gen_length_comparison(axes[0], correct_only=True)

	print("\nGenerating plot for incorrect answers with repetitions...")
	plot_incorrect_repetition_distribution(axes[1])

	print("\nGenerating plot for LCS distribution...")
	plot_lcs_distribution(axes[2])

	plt.subplots_adjust(top=0.92, bottom=0.2, left=0.06, right=0.98)
	fig.savefig(OUT_PTH_COMBINED, dpi=200)
	print(f"\nCombined plot saved to: {OUT_PTH_COMBINED}")


if __name__ == "__main__":
	main()
