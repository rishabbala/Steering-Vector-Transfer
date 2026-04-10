#!/usr/bin/env python3
"""
For each of 3 base models and 3 datasets:
1) Find the best test setting across:
   - /projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_avg
   - /projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_pca
   using max acc from *_metrics.json across all hyperparameters.
2) For that best setting, bin answers by len(code[0]) < 50 vs >= 50.
3) Plot, for each (model, dataset), two stacked bars (<50 and >=50) whose total height
   is the % of answers in that bin, with correct vs incorrect stacked inside.

Output: one figure containing a 3x3 grid (rows=models, cols=datasets).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import matplotlib.font_manager as fm

import os

BASE_DIR = Path("/projects/llms-lab/transfer_compare")
AVG_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_avg"
PCA_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_pca"
BASE_EVAL_DIR = BASE_DIR / "base_evals_test"

DATASETS = ["math", "gsm8k", "svamp"]

# The 3 models you’ve been using (match by base model name: first token before "_+_")
TARGET_BASE_MODELS: Dict[str, str] = {
	# "Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": "Qwen-1.5-14B",
	# "gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": "gemma-2-9B",
	# "OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": "OLMo-2-7B",
	"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": "Qwen-1.5-7B",
	"gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": "gemma-2-2B",
	"OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-1B",
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


@dataclass(frozen=True)
class BestConfig:
	method: str  # "avg" | "pca"
	base_model: str
	model_display: str
	dataset: str
	acc: float
	num_train_samples: int
	rank: int
	metrics_path: Path
	jsonl_path: Path


def _parse_num_train_samples(part: str) -> Optional[int]:
	m = re.match(r"^num_train_samples_(\d+)$", part)
	if not m:
		return None
	try:
		return int(m.group(1))
	except Exception:
		return None


def _parse_rank(part: str) -> Optional[int]:
	m = re.match(r"^rank_(\d+)$", part)
	if not m:
		return None
	try:
		return int(m.group(1))
	except Exception:
		return None


def _find_jsonl_for_metrics(metrics_path: Path) -> Optional[Path]:
	guess = metrics_path.with_name(metrics_path.name.replace("_metrics.json", ".jsonl"))
	if guess.exists():
		return guess
	cands = list(metrics_path.parent.glob("*.jsonl"))
	return cands[0] if cands else None


def _find_baseline_jsonl(base_model: str, dataset: str, prompt: str) -> Optional[Path]:
	"""
	Base evals live under:
	  base_evals_test/<base_model>/<dataset>/*_prompt_general-{direct|cot}.jsonl
	"""
	d = BASE_EVAL_DIR / base_model / dataset
	if not d.exists():
		return None
	pattern = f"*_prompt_general-{prompt}.jsonl"
	cands = sorted(d.glob(pattern))
	if cands:
		return cands[0]
	# Fallback: any jsonl mentioning prompt type, excluding metrics
	cands = sorted(
		p for p in d.glob("*.jsonl") if "metrics" not in p.name and prompt in p.name
	)
	return cands[0] if cands else None


def _scan_best_configs(root: Path, method: str) -> Dict[Tuple[str, str], BestConfig]:
	"""
	Return best config per (base_model, dataset) from a given root directory.
	"""
	best: Dict[Tuple[str, str], BestConfig] = {}
	if not root.exists():
		print(f"Warning: missing dir: {root}")
		return best

	for metrics_path in root.rglob("*_metrics.json"):
		dataset = metrics_path.parent.name
		if dataset not in DATASETS:
			continue

		parts = metrics_path.parts

		nts_idx = None
		num_train_samples = None
		for i, p in enumerate(parts):
			nts = _parse_num_train_samples(p)
			if nts is not None:
				nts_idx = i
				num_train_samples = nts
				break
		if nts_idx is None or num_train_samples is None:
			continue
		if nts_idx + 2 >= len(parts):
			continue

		model_family = parts[nts_idx + 1]
		if model_family not in TARGET_BASE_MODELS:
			continue

		rank = None
		for j in range(nts_idx + 2, min(nts_idx + 7, len(parts))):
			r = _parse_rank(parts[j])
			if r is not None:
				rank = r
				break
		if rank is None:
			continue

		try:
			with metrics_path.open("r") as f:
				metrics = json.load(f)
			acc = float(metrics.get("acc", 0.0))
		except Exception:
			continue

		jsonl_path = _find_jsonl_for_metrics(metrics_path)
		if jsonl_path is None:
			continue

		key = (model_family, dataset)
		cfg = BestConfig(
			method=method,
			base_model=model_family,
			model_display=TARGET_BASE_MODELS[model_family],
			dataset=dataset,
			acc=acc,
			num_train_samples=num_train_samples,
			rank=rank,
			metrics_path=metrics_path,
			jsonl_path=jsonl_path,
		)
		if key not in best or cfg.acc > best[key].acc:
			best[key] = cfg

	return best


def find_best_across_avg_pca() -> Dict[Tuple[str, str], BestConfig]:
	best_avg = _scan_best_configs(AVG_DIR, "avg")
	best_pca = _scan_best_configs(PCA_DIR, "pca")

	out: Dict[Tuple[str, str], BestConfig] = {}
	for key in set(best_avg.keys()) | set(best_pca.keys()):
		a = best_avg.get(key)
		p = best_pca.get(key)
		if a is None:
			out[key] = p
		elif p is None:
			out[key] = a
		else:
			out[key] = a if a.acc >= p.acc else p
	return out


def compute_bin_stats(
	jsonl_path: Path, threshold: int = 50
) -> Optional[Dict[str, float]]:
	"""
	Returns percentages out of total counted answers (rows that have code[0]):
	- short_total_pct, long_total_pct
	- short_correct_pct, short_incorrect_pct
	- long_correct_pct, long_incorrect_pct
	"""
	short_total = 0
	long_total = 0
	short_correct = 0
	long_correct = 0

	try:
		with jsonl_path.open("r") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					row = json.loads(line)
				except json.JSONDecodeError:
					continue
				code = row.get("code")
				if not isinstance(code, list) or not code:
					continue
				code0 = code[0]
				if not isinstance(code0, str):
					continue
				L = len(code0)

				score = row.get("score")
				is_correct = False
				if isinstance(score, list) and score:
					is_correct = bool(score[0])

				if L < threshold:
					short_total += 1
					if is_correct:
						short_correct += 1
				else:
					long_total += 1
					if is_correct:
						long_correct += 1
	except FileNotFoundError:
		return None
	except Exception:
		return None

	total = short_total + long_total
	if total == 0:
		return None

	short_incorrect = short_total - short_correct
	long_incorrect = long_total - long_correct

	return {
		"short_total_pct": 100.0 * short_total / total,
		"long_total_pct": 100.0 * long_total / total,
		"short_correct_pct": 100.0 * short_correct / total,
		"short_incorrect_pct": 100.0 * short_incorrect / total,
		"long_correct_pct": 100.0 * long_correct / total,
		"long_incorrect_pct": 100.0 * long_incorrect / total,
		"n_total": float(total),
	}


def plot_one_figure(
	best: Dict[Tuple[str, str], BestConfig], threshold: int = 50
) -> Path:
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
			"axes.labelsize": 30,
			"xtick.labelsize": 15,
			"axes.labelweight": "bold",
			"ytick.labelsize": 15,
			"legend.fontsize": 15,
			"hatch.linewidth": 0.5,
		}
	)
	models = list(TARGET_BASE_MODELS.keys())
	datasets = DATASETS

	fig, axes = plt.subplots(
		nrows=len(models),
		ncols=len(datasets),
		figsize=(18, 12),
		sharey=True,
	)

	# Slightly darker pastel colors + subtle hatch shading
	c_correct = "#63B78A"  # darker pastel green
	c_incorrect = "#E67676"  # darker pastel red
	edge = "#000000"
	hatch_correct = "..."
	hatch_incorrect = "///"

	for r, base_model in enumerate(models):
		for c, dataset in enumerate(datasets):
			ax = axes[r, c]
			key = (base_model, dataset)
			cfg = best.get(key)
			# Gather 3 sources: Base-Direct, Base-CoT, and best Steered (avg/pca).
			stats_direct = None
			stats_cot = None
			stats_steered = None

			direct_path = _find_baseline_jsonl(
				base_model.split("_+_")[0], dataset, "direct"
			)
			if direct_path:
				stats_direct = compute_bin_stats(direct_path, threshold=threshold)

			cot_path = _find_baseline_jsonl(base_model.split("_+_")[0], dataset, "cot")
			if cot_path:
				stats_cot = compute_bin_stats(cot_path, threshold=threshold)

			if cfg is not None:
				stats_steered = compute_bin_stats(cfg.jsonl_path, threshold=threshold)

			sources = [
				(rf"$\mathcal{{T}}_\mathrm{{L}}$" + "\nDirect", stats_direct),
				(rf"$\mathcal{{T}}_\mathrm{{L}}$" + "\nCoT", stats_cot),
				(rf"$\mathcal{{T}}_\mathrm{{U}}$" + "\nDirect", stats_steered),
			]
			if all(s is None for _, s in sources):
				ax.text(
					0.5,
					0.5,
					"no data",
					ha="center",
					va="center",
					transform=ax.transAxes,
					# fontsize=15,
				)
				ax.set_ylim(0, 100)
				continue

			# 6 bars: (Direct short/long), (CoT short/long), (Steered short/long)
			# each bar stacked correct/incorrect, where bar height is % of answers in that bin.
			positions = []
			labels = []
			short_positions = {}
			long_positions = {}
			x = 0
			for name, st in sources:
				short_positions[name] = x
				long_positions[name] = x + 1
				positions.extend([x, x + 1])
				labels.extend([f"{name}\n<{threshold}", f"{name}\n≥{threshold}"])
				x += 2

			for name, st in sources:
				if st is None:
					continue
				sc = st["short_correct_pct"]
				si = st["short_incorrect_pct"]
				lc = st["long_correct_pct"]
				li = st["long_incorrect_pct"]

				xs = short_positions[name]
				xl = long_positions[name]

				ax.bar(
					xs,
					sc,
					color=c_correct,
					hatch=hatch_correct,
					edgecolor=edge,
					linewidth=0.5,
					label="Correct"
					if (r == 0 and c == 0 and name == "Direct")
					else None,
				)
				ax.bar(
					xs,
					si,
					bottom=sc,
					color=c_incorrect,
					hatch=hatch_incorrect,
					edgecolor=edge,
					linewidth=0.5,
					label="Incorrect"
					if (r == 0 and c == 0 and name == "Direct")
					else None,
				)
				ax.bar(
					xl,
					lc,
					color=c_correct,
					hatch=hatch_correct,
					edgecolor=edge,
					linewidth=0.5,
				)
				ax.bar(
					xl,
					li,
					bottom=lc,
					color=c_incorrect,
					hatch=hatch_incorrect,
					edgecolor=edge,
					linewidth=0.5,
				)

			ax.set_xticks(positions)
			ax.set_xticklabels(labels)
			ax.set_ylim(0, 100)
			ax.grid(axis="y")
			ax.set_axisbelow(True)

			if c == 0 and r == 1:
				# Requested y-label wording
				ax.set_ylabel("% of correct answers", fontsize=30, fontweight="bold")

			# Removed config annotations (acc/rank/ntrain) per request

	# fig.suptitle(
	# 	f"Length bin (<{threshold} vs ≥{threshold}) × Correctness (Base Direct/CoT + best Steered)",
	# 	y=0.995,
	# 	fontweight="bold",
	# )
	# Always-show legend (don't rely on bar labels being present)
	legend_handles = [
		Patch(
			facecolor=c_correct, edgecolor=edge, hatch=hatch_correct, label="Correct"
		),
		Patch(
			facecolor=c_incorrect,
			edgecolor=edge,
			hatch=hatch_incorrect,
			label="Incorrect",
		),
	]
	fig.legend(
		handles=legend_handles,
		loc="upper right",
		bbox_to_anchor=(1, 0.66),
		framealpha=0.95,
	)
	# Ensure enough vertical space between rows so row titles don't collide with the
	# previous row's x tick labels.
	fig.subplots_adjust(hspace=0.75, wspace=0.1)

	# Column titles (datasets) at the top of the figure.
	# Place them using the top row axes positions so they align with columns.
	for c, dataset in enumerate(datasets):
		ax = axes[0, c]
		ax.text(
			0.5,
			1.3,
			dataset.upper(),
			ha="center",
			va="bottom",
			fontsize=30,
			fontweight="bold",
			transform=ax.transAxes,
		)

	# Row titles: paper name centered above each row of subplots.
	# Do this after tight_layout so axes positions are final.
	for r, base_model in enumerate(models):
		paper = PAPER_NAME.get(base_model, base_model)
		left = axes[r, 0].get_position()
		right = axes[r, -1].get_position()
		x_center = 0.5 * (left.x0 + right.x1)
		# A slightly larger offset keeps the title in the inter-row gap.
		y = left.y1 + 0.03
		fig.text(
			x_center, y, paper, ha="center", va="bottom", fontsize=25, fontweight="bold"
		)

	plt.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.1)

	out_path = Path("figures/len_bins_large2small.pdf")
	fig.savefig(out_path)
	plt.close(fig)
	return out_path


def main():
	print("Scanning best configs from test avg/pca...")
	best = find_best_across_avg_pca()
	print(f"Best configs found: {len(best)} (model×dataset)")

	out = plot_one_figure(best, threshold=50)
	print(f"Saved: {out}")


if __name__ == "__main__":
	main()
