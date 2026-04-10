#!/usr/bin/env python3
"""Plot accuracy comparison: Direct baseline vs Best Transfer (avg/pca)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib as mp

import matplotlib.font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fm.fontManager.addfont(
# 	os.path.expanduser("~/.fonts/inconsolata/fonts/ttf/Inconsolata-Regular.ttf")
# )
# # disable usetex
# mp.rcParams.update(
# 	{
# 		"text.usetex": False,
# 		"font.family": "monospace",  # use monospace for all text
# 		"font.monospace": ["Inconsolata"],  # set Inconsolata as the monospace font
# 		"mathtext.fontset": "dejavuserif",
# 	}
# )


# mp.rcParams.update(
# 	{
# 		"text.usetex": False,
# 		"font.family": "monospace",
# 		"font.monospace": ["DejaVu Sans Mono"],
# 		"mathtext.fontset": "dejavusans",
# 	}
# )


# mp.rcParams.update(
# 	{
# 		"font.family": "serif",
# 		"font.serif": ["Palatino", "Palatino Linotype", "TeX Gyre Pagella", "serif"],
# 	}
# )

# ----------------------------
# Paths / constants
# ----------------------------
BASE_DIR = Path("/projects/llms-lab/transfer_compare")
OOD_AVG_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_avg"
OOD_PCA_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_pca"
BASE_EVAL_DIR = BASE_DIR / "base_evals_test"

DATASETS: List[str] = ["gsm8k", "math", "svamp"]
FIGURES_DIR = Path("/home/rishbb/Qwen2.5-Math/figures")
MODEL = "Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B"
OUTPUT_PATH = FIGURES_DIR / "plot_cot.pdf"

PAPER_NAME = {
	"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": "Qwen1.5-7B + {to_small_caps('unlock')}_{ from 7B}",
}

PAPER_DATASET_NAME = {
	"gsm8k": "GSM8K\n",
	"math": "MATH\n",
	"svamp": "SVAMP\n",
}


# ----------------------------
# Styling helpers
# ----------------------------
def _apply_plot_style() -> None:
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
			"xtick.labelsize": 20,
			"axes.labelweight": "bold",
			"ytick.labelsize": 20,
			"legend.fontsize": 18,
			"hatch.linewidth": 0.5,
		}
	)


# ----------------------------
# Data model
# ----------------------------
@dataclass(frozen=True)
class TransferBest:
	method: str
	model_family: str
	dataset: str
	acc: float
	num_train_samples: Optional[int]
	rank: Optional[int]
	alpha: Optional[float]
	metrics_path: Path


# ----------------------------
# Parsing / scanning
# ----------------------------
_RE_NUM_TRAIN = re.compile(r"^num_train_samples_(\d+)$")
_RE_RANK = re.compile(r"^rank_(\d+)$")
_RE_ALPHA = re.compile(r"alpha_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _read_json(path: Path) -> Optional[dict]:
	try:
		with open(path, "r") as f:
			return json.load(f)
	except Exception:
		return None


def _read_acc(metrics_path: Path) -> Optional[float]:
	data = _read_json(metrics_path)
	if not data:
		return None
	acc = data.get("acc", None)
	if acc is None:
		return None
	try:
		return float(acc)
	except Exception:
		return None


def _parse_alpha_from_name(name: str) -> Optional[float]:
	m = _RE_ALPHA.search(name)
	if not m:
		return None
	try:
		return float(m.group(1))
	except Exception:
		return None


def scan_best_transfer(
	root: Path,
	method: str,
	model_allowlist: Optional[set[str]] = None,
) -> Dict[Tuple[str, str], TransferBest]:
	best: Dict[Tuple[str, str], TransferBest] = {}
	if not root.exists():
		return best

	for nts_dir in root.iterdir():
		m_nts = _RE_NUM_TRAIN.match(nts_dir.name) if nts_dir.is_dir() else None
		if not m_nts:
			continue
		try:
			num_train_samples = int(m_nts.group(1))
		except Exception:
			num_train_samples = None

		for model_dir in nts_dir.iterdir():
			if not model_dir.is_dir():
				continue
			model_family = model_dir.name
			if model_allowlist is not None and model_family not in model_allowlist:
				continue

			for rank_dir in model_dir.iterdir():
				m_rank = _RE_RANK.match(rank_dir.name) if rank_dir.is_dir() else None
				if not m_rank:
					continue
				try:
					rank = int(m_rank.group(1))
				except Exception:
					rank = None

				for steer_dir in rank_dir.iterdir():
					if not steer_dir.is_dir():
						continue
					for dataset_dir in steer_dir.iterdir():
						if not dataset_dir.is_dir():
							continue
						dataset = dataset_dir.name
						if dataset not in DATASETS:
							continue

						for metrics_path in dataset_dir.glob("*_metrics.json"):
							acc = _read_acc(metrics_path)
							if acc is None:
								continue
							alpha = _parse_alpha_from_name(metrics_path.name)

							key = (model_family, dataset)
							info = TransferBest(
								method=method,
								model_family=model_family,
								dataset=dataset,
								acc=acc,
								num_train_samples=num_train_samples,
								rank=rank,
								alpha=alpha,
								metrics_path=metrics_path,
							)
							prev = best.get(key)
							if prev is None or info.acc > prev.acc:
								best[key] = info

	return best


def combine_best(*best_dicts) -> Dict[Tuple[str, str], TransferBest]:
	all_keys = set()
	for d in best_dicts:
		all_keys |= set(d.keys())

	out: Dict[Tuple[str, str], TransferBest] = {}
	for k in all_keys:
		candidates = [d.get(k) for d in best_dicts if d.get(k) is not None]
		if candidates:
			out[k] = max(candidates, key=lambda x: x.acc)
	return out


# ----------------------------
# Baseline helpers
# ----------------------------
def extract_base_model_name(model_family: str) -> str:
	return model_family.split("_+_")[0]


def get_instruct_model_name(base_model: str) -> str:
	if "Qwen1.5" in base_model:
		return base_model + "-Chat"
	return base_model


def _find_baseline_direct_metrics(base_model: str, dataset: str) -> Optional[Path]:
	d = BASE_EVAL_DIR / base_model / dataset
	if not d.exists():
		return None
	cands = sorted(d.glob("*_prompt_general-direct_metrics.json"))
	if cands:
		return cands[0]
	cands = sorted(p for p in d.glob("*metrics.json") if "direct" in p.name)
	return cands[0] if cands else None


def _find_baseline_cot_metrics(base_model: str, dataset: str) -> Optional[Path]:
	d = BASE_EVAL_DIR / base_model / dataset
	if not d.exists():
		return None
	cands = sorted(d.glob("*_prompt_general-cot_metrics.json"))
	return cands[0] if cands else None


def read_direct_acc_for_model_family(
	model_family: str, dataset: str
) -> Optional[float]:
	base_model = extract_base_model_name(model_family)
	metrics_path = _find_baseline_direct_metrics(base_model, dataset)
	if metrics_path is None:
		return None
	return _read_acc(metrics_path)


def read_cot_acc_for_model_family(model_family: str, dataset: str) -> Optional[float]:
	base_model = extract_base_model_name(model_family)
	metrics_path = _find_baseline_cot_metrics(base_model, dataset)
	if metrics_path is None:
		return None
	return _read_acc(metrics_path)


def read_instruct_cot_acc_for_model_family(
	model_family: str, dataset: str
) -> Optional[float]:
	base_model = extract_base_model_name(model_family)
	instruct_model = get_instruct_model_name(base_model)
	metrics_path = _find_baseline_cot_metrics(instruct_model, dataset)
	if metrics_path is None:
		return None
	return _read_acc(metrics_path)


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


# ----------------------------
# Main plotting
# ----------------------------
def plot_accuracy(best_transfer: Dict[Tuple[str, str], TransferBest]) -> None:
	_apply_plot_style()

	model = MODEL
	base_model_name = extract_base_model_name(model)
	instruct_model_name = get_instruct_model_name(base_model_name)

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	n_datasets = len(DATASETS)
	bar_width = 0.25
	x = np.arange(n_datasets)

	base_vals = []
	instruct_vals = []
	best_vals = []

	print("Best OOD files per dataset:")
	for dataset in DATASETS:
		dacc = read_direct_acc_for_model_family(base_model_name, dataset)
		base_vals.append(float(dacc) if dacc is not None else 0.0)

		cot_instruct = read_cot_acc_for_model_family(instruct_model_name, dataset)
		instruct_vals.append(float(cot_instruct) if cot_instruct is not None else 0.0)

		bt = best_transfer.get((model, dataset))
		best_vals.append(float(bt.acc) if bt is not None else 0.0)

		if bt is not None:
			print(f"  {dataset}: {bt.metrics_path}")
		else:
			print(f"  {dataset}: no data")

	# color_palette = ["#D6EAF8", "#3498DB", "#1B4F72", "#D55E00", "#CC79A7"]
	color_palette = [
		"#FBF3E9",  # bright icy blue
		"#EECA9B",  # bright sky blue
		"#D98C25",  # cobalt ocean blue
	]

	ax.bar(
		x - bar_width,
		base_vals,
		bar_width,
		label=r"Qwen1.5-7B w/o CoT",
		color=color_palette[0],
		edgecolor="black",
		linewidth=1,
	)
	ax.bar(
		x,
		instruct_vals,
		bar_width,
		label=rf"Qwen1.5-7B-Chat w/ CoT",
		color=color_palette[1],
		edgecolor="black",
		linewidth=1,
	)
	ax.bar(
		x + bar_width,
		best_vals,
		bar_width,
		label="Qwen1.5-7B"
		+ f" + {to_small_caps('unlock')}$_{{ \mathrm{{from\;14B}}}}$ w/o CoT",
		color=color_palette[2],
		edgecolor="black",
		linewidth=1,
	)

	# ax.set_xlabel("Dataset")
	ax.set_ylabel("Accuracy (%)")
	# ax.set_title(
	# 	f"{PAPER_NAME.get(model, base_model_name)}",
	# )
	ax.set_xticks(x)
	ax.set_xticklabels(
		[PAPER_DATASET_NAME.get(d, d) for d in DATASETS], fontweight="bold"
	)
	ax.legend(loc="upper left")
	ax.grid(axis="y", alpha=0.3)
	ax.set_axisbelow(True)
	ax.set_ylim(0, 100)

	# for label in ax.get_xticklabels():
	# 	label.set_fontweight("bold")

	plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.13)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUTPUT_PATH)
	plt.close(fig)


def main() -> None:
	allowlist = {MODEL}

	ood_avg_best = scan_best_transfer(OOD_AVG_DIR, "ood_avg", model_allowlist=allowlist)
	ood_pca_best = scan_best_transfer(OOD_PCA_DIR, "ood_pca", model_allowlist=allowlist)

	best_transfer = combine_best(ood_avg_best, ood_pca_best)

	plot_accuracy(best_transfer)


if __name__ == "__main__":
	os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
	main()
