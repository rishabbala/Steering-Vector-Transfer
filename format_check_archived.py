import json
import os
import string
from collections import Counter

import matplotlib.pyplot as plt


# ROOT_DIR = "/projects/llms-lab/transfer_compare"
IN_PTH = [
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca/num_train_samples_512/Qwen3-14B-Base_+_Qwen3-4B_-_Qwen3-4B-Base/rank_512/pca_math_steering_math/agieval_math/Qwen_Qwen3-14B-Base_alpha_0.05_test_num_samples_-1_max_gen_tokens_4096.jsonl",
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca/num_train_samples_512/Qwen3-14B-Base_+_Qwen3-4B_-_Qwen3-4B-Base/rank_512/pca_math_steering_math/deepmind_math/Qwen_Qwen3-14B-Base_alpha_0.05_test_num_samples_-1_max_gen_tokens_4096.jsonl",
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca/num_train_samples_512/Qwen3-14B-Base_+_Qwen3-4B_-_Qwen3-4B-Base/rank_4/pca_math_steering_math/minerva_math/Qwen_Qwen3-14B-Base_alpha_0.05_test_num_samples_-1_max_gen_tokens_4096.jsonl",
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca/num_train_samples_512/Qwen3-14B-Base_+_Qwen3-4B_-_Qwen3-4B-Base/rank_4/pca_math_steering_math/olympiadbench/Qwen_Qwen3-14B-Base_alpha_0.05_test_num_samples_-1_max_gen_tokens_4096.jsonl",
]
MODEL_NAME = IN_PTH[0].split("num_train_samples")[1].split("/")[1]

OUT_PTH = f"./figures/{MODEL_NAME}_format_check.png"

if not os.path.exists("./figures"):
	os.makedirs("./figures")


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


def _normalize_token(token: str) -> str:
	token = token.strip()
	token = token.strip(string.punctuation)
	return token.lower()


def _get_code_text(entry) -> str:
	code = entry.get("code")
	if isinstance(code, list):
		return "\n".join(str(part) for part in code)
	return str(code) if code is not None else ""


def _first_two_tokens(text: str):
	tokens = text.split()
	normalized = []
	for token in tokens:
		token = _normalize_token(token)
		if token:
			normalized.append(token)
		if len(normalized) >= 2:
			break
	first_word = normalized[0] if len(normalized) >= 1 else None
	second_word = normalized[1] if len(normalized) >= 2 else None
	return first_word, second_word


def _percentage_distribution(counter: Counter, total: int):
	if total == 0:
		return []
	return sorted(
		[(word, (count / total) * 100) for word, count in counter.items()],
		key=lambda item: item[1],
		reverse=True,
	)


def get_format_percentage(file_path):
	data = []
	with open(file_path, "r") as file:
		for line in file:
			data.append(json.loads(line))

	ending_strings = ["The final answer is <atok>"]
	count_with_ending = 0
	total = len(data)

	for d in data:
		code_text = d["code"][0]
		if any(x in code_text for x in ending_strings):
			count_with_ending += 1

	percentage = (count_with_ending / total * 100) if total > 0 else 0
	return percentage, count_with_ending, total


def plot_format_comparison():
	model_names = []
	correct_pcts = []
	incorrect_pcts = []

	for steered_path in IN_PTH:
		dataset_name = steered_path.split("steering_")[-1].split("/")[1]

		steered_pct, steered_count, steered_total = get_format_percentage(steered_path)
		model_names.append(f"{dataset_name} (Steered)")
		correct_pcts.append(steered_pct)
		incorrect_pcts.append(100 - steered_pct)

		base_path = get_base_path(steered_path)
		if base_path and os.path.exists(base_path):
			base_pct, base_count, base_total = get_format_percentage(base_path)
			model_names.append(f"{dataset_name} (Base)")
			correct_pcts.append(base_pct)
			incorrect_pcts.append(100 - base_pct)

			print(
				f"{dataset_name} - Steered: {steered_count}/{steered_total} ({steered_pct:.2f}%)"
			)
			print(f"{dataset_name} - Base: {base_count}/{base_total} ({base_pct:.2f}%)")
		else:
			print(f"Base path not found: {base_path}")

	num_bars = len(model_names)
	fig_height = max(8, num_bars * 0.6)
	fig, ax = plt.subplots(figsize=(14, fig_height))

	y_pos = range(num_bars)

	c_correct = "#1f77b4"
	c_incorrect = "#d62728"

	ax.barh(y_pos, correct_pcts, color=c_correct, label="Correct Format")
	ax.barh(
		y_pos,
		incorrect_pcts,
		left=correct_pcts,
		color=c_incorrect,
		label="Incorrect Format",
	)

	ax.set_yticks(y_pos)
	ax.set_yticklabels(model_names, fontsize=12)
	ax.set_xlabel("Percentage (%)", fontsize=14)
	ax.set_title("Format Correctness Comparison", fontsize=16, fontweight="bold")
	ax.set_xlim(0, 100)
	ax.legend(fontsize=12, loc="lower right")

	for i, (correct, incorrect) in enumerate(zip(correct_pcts, incorrect_pcts)):
		if correct > 5:
			ax.text(
				correct / 2,
				i,
				f"{correct:.1f}%",
				ha="center",
				va="center",
				fontsize=10,
				fontweight="bold",
				color="white",
			)
		if incorrect > 5:
			ax.text(
				correct + incorrect / 2,
				i,
				f"{incorrect:.1f}%",
				ha="center",
				va="center",
				fontsize=10,
				fontweight="bold",
				color="white",
			)

	plt.tight_layout()
	fig.savefig(OUT_PTH, dpi=200, bbox_inches="tight")
	print(f"\nPlot saved to: {OUT_PTH}")


def main():
	plot_format_comparison()


if __name__ == "__main__":
	main()
