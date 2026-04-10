import json
import os
import string
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


IN_PTH = "/projects/llms-lab/transfer_compare/base_evals_test/Qwen2.5-14B/olympiadbench/Qwen_Qwen2.5-14B_test_num_samples_-1_max_gen_tokens_4096_prompt_general-cot.jsonl"
if "num_train_samples_" in IN_PTH:
	MODEL_NAME = IN_PTH.split("num_train_samples")[1].split("/")[1]
	DATASET_NAME = IN_PTH.split("steering_")[-1].split("/")[1]
else:
	MODEL_NAME = (
		IN_PTH.split("/")[-1].split("_test_num_samples_-1")[0].split("_alpha_")[0]
	)
	DATASET_NAME = IN_PTH.split("/")[6]

PAPER_DATASET_NAME = {
	"agieval_math": "Agieval Math",
	"minerva_math": "Minerva Math",
	"olympiadbench": "Olympiad Bench",
	"deepmind_math": "Deepmind Math",
}

print(MODEL_NAME, DATASET_NAME)
OUT_PTH = f"./figures/word_cloud/{MODEL_NAME}_{DATASET_NAME}_word_distribution.pdf"

if not os.path.exists("./figures/word_cloud"):
	os.makedirs("./figures/word_cloud")


def _percentage_distribution(counter: Counter, total: int):
	if total == 0:
		return []
	return sorted(
		[(word, (count / total) * 100) for word, count in counter.items()],
		key=lambda item: item[1],
		reverse=True,
	)


def build_word_distributions():
	data = []
	with open(IN_PTH, "r") as file:
		for line in file:
			data.append(json.loads(line))

	output_txt = []
	for d in data:
		words = d["code"][0].split(" ")
		if len(words) < 2:
			continue
		output_txt.append(
			[
				words[0][:5],
				words[1],
			]
		)

	first, second = zip(*output_txt)

	first_word = []
	second_word = []
	for i in range(len(first)):
		first_word.append(first[i])
		second_word.append(second[i])

	plot_word_distributions(first_word, second_word)


def plot_word_distributions(first_word, second_word):
	first_counter = Counter(first_word)
	second_counter = Counter(second_word)

	total_first = sum(first_counter.values())
	total_second = sum(second_counter.values())

	first_dist = _percentage_distribution(first_counter, total_first)
	second_dist = _percentage_distribution(second_counter, total_second)

	first_dist = first_dist[:5]
	second_dist = second_dist[:5]

	first_words, first_vals = zip(*first_dist) if first_dist else ([], [])
	first_words = [word.replace("$", "\\$") for word in first_words]

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
			"axes.titlesize": 40,
			"axes.titleweight": "bold",
			"axes.labelsize": 38,
			"xtick.labelsize": 35,
			"axes.labelweight": "bold",
			"ytick.labelsize": 35,
			"legend.fontsize": 32,
			"hatch.linewidth": 0.5,
		}
	)

	fig, axes = plt.subplots(1, 1, figsize=(8, 8))
	plt.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.15)

	axes.barh(first_words, first_vals, color="#0072B2")
	axes.invert_yaxis()
	axes.set_title(f"{PAPER_DATASET_NAME[DATASET_NAME]}")
	axes.set_xlabel("Percentage")
	axes.tick_params(axis="both")
	fig.savefig(OUT_PTH, dpi=1000)


def main():
	build_word_distributions()


if __name__ == "__main__":
	main()
