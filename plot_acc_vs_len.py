#!/usr/bin/env python3
"""
Scatter plots: accuracy vs average generation length (all answers).

For each model (3 total), generate a 2x3 grid:
- rows: avg, pca
- cols: math, gsm8k, svamp

Each point is a hyperparameter configuration (rank, num_train_samples):
- x = mean(len(code[0])) over ALL rows in the jsonl (ignore correctness)
- y = acc from *_metrics.json
- color = rank
- size = num_train_samples
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

BASE_DIR = Path("/projects/llms-lab/transfer_compare")
AVG_DIR = BASE_DIR / "hs_svd_arch_transfer_cot_avg_grid_search"
PCA_DIR = BASE_DIR / "hs_svd_arch_transfer_cot_pca_grid_search"

DATASETS = ["math", "gsm8k", "svamp"]

# Only these 3 base models (match the first model name before "_+_")
TARGET_BASE_MODELS: Dict[str, str] = {
	"Qwen1.5-14B": "Qwen-1.5-14B",
	"gemma-2-9b": "Gemma-2-9B",
	"OLMo-2-1124-7B": "OLMo-2-7B",
}


@dataclass(frozen=True)
class ConfigPoint:
	method: str  # "avg" | "pca"
	dataset: str  # math | gsm8k | svamp
	model_base: str
	model_display: str
	num_train_examples: int
	rank: int
	acc: float  # metrics acc (likely percent)
	metrics_path: Path
	jsonl_path: Path


def _apply_style():
	plt.rcParams.update(
		{
			"figure.dpi": 140,
			"savefig.dpi": 300,
			"axes.grid": True,
			"grid.alpha": 0.28,
			"grid.linestyle": "--",
			"grid.linewidth": 0.8,
			"axes.titleweight": "bold",
			"axes.labelweight": "bold",
		}
	)


def _safe_int(s: str) -> Optional[int]:
	try:
		return int(s)
	except Exception:
		return None


def _parse_num_train_examples(part: str) -> Optional[int]:
	# part like "num_train_examples_128" (preferred) or legacy "num_train_samples_128"
	m = re.match(r"^(?:num_train_examples|num_train_samples)_(\d+)$", part)
	if not m:
		return None
	return _safe_int(m.group(1))


def _parse_rank(part: str) -> Optional[int]:
	# part like "rank_64"
	m = re.match(r"^rank_(\d+)$", part)
	if not m:
		return None
	return _safe_int(m.group(1))


def _mean_code0_len(jsonl_path: Path) -> Optional[float]:
	total = 0
	n = 0
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
				if not isinstance(code0, str) or not code0:
					continue
				total += len(code0)
				n += 1
	except FileNotFoundError:
		return None
	except Exception:
		return None

	if n == 0:
		return None
	return total / n


def _scan_method_root(method: str, root: Path) -> List[ConfigPoint]:
	points: List[ConfigPoint] = []
	if not root.exists():
		print(f"Warning: missing dir: {root}")
		return points

	for metrics_path in root.rglob("*_metrics.json"):
		# Expected structure:
		# .../num_train_samples_<N>/<model_family>/rank_<R>/<steer>/<dataset>/<file>_metrics.json
		dataset = metrics_path.parent.name
		if dataset not in DATASETS:
			continue

		parts = metrics_path.parts

		# Find num_train_samples_*
		nts_idx = None
		num_train_examples = None
		for i, p in enumerate(parts):
			nts = _parse_num_train_examples(p)
			if nts is not None:
				nts_idx = i
				num_train_examples = nts
				break
		if nts_idx is None or num_train_examples is None:
			continue

		# Model family should follow num_train_samples_*
		if nts_idx + 2 >= len(parts):
			continue
		model_family = parts[nts_idx + 1]
		model_base = model_family.split("_+_")[0]
		if model_base not in TARGET_BASE_MODELS:
			continue

		# Rank should be somewhere after model family; typically next component.
		rank = None
		for j in range(nts_idx + 2, min(nts_idx + 6, len(parts))):
			r = _parse_rank(parts[j])
			if r is not None:
				rank = r
				break
		if rank is None:
			continue

		# Read acc
		try:
			with metrics_path.open("r") as f:
				metrics = json.load(f)
			acc = float(metrics.get("acc", 0.0))
		except Exception:
			continue

		# Match jsonl: metrics file name with _metrics.json removed
		jsonl_path = metrics_path.with_name(
			metrics_path.name.replace("_metrics.json", ".jsonl")
		)
		if not jsonl_path.exists():
			# fallback: pick any jsonl in same folder
			cands = list(metrics_path.parent.glob("*.jsonl"))
			if not cands:
				continue
			jsonl_path = cands[0]

		points.append(
			ConfigPoint(
				method=method,
				dataset=dataset,
				model_base=model_base,
				model_display=TARGET_BASE_MODELS[model_base],
				num_train_examples=num_train_examples,
				rank=rank,
				acc=acc,
				metrics_path=metrics_path,
				jsonl_path=jsonl_path,
			)
		)

	return points


def scan_all_points() -> List[ConfigPoint]:
	points: List[ConfigPoint] = []
	points.extend(_scan_method_root("avg", AVG_DIR))
	points.extend(_scan_method_root("pca", PCA_DIR))
	return points


def _sizes_from_ntrain(
	ntrains: List[int], min_s: float = 35, max_s: float = 220
) -> Dict[int, float]:
	# Use log scaling so 16..1024 sizes are readable
	if not ntrains:
		return {}
	uniq = sorted(set(ntrains))
	if len(uniq) == 1:
		return {uniq[0]: (min_s + max_s) / 2}
	logs = [math.log2(x) for x in uniq]
	lo, hi = min(logs), max(logs)
	out: Dict[int, float] = {}
	for n, lg in zip(uniq, logs):
		t = 0.0 if hi == lo else (lg - lo) / (hi - lo)
		out[n] = min_s + t * (max_s - min_s)
	return out


def _sanitize_filename(s: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def plot_for_model(model_base: str, points: List[ConfigPoint]) -> None:
	model_display = TARGET_BASE_MODELS[model_base]
	model_points = [p for p in points if p.model_base == model_base]
	if not model_points:
		print(f"[{model_display}] no points found; skipping")
		return

	# Precompute x = mean len(code[0]) for each config
	x_mean: Dict[Path, float] = {}
	for p in model_points:
		if p.jsonl_path not in x_mean:
			val = _mean_code0_len(p.jsonl_path)
			if val is not None:
				x_mean[p.jsonl_path] = val

	# Keep only those with x
	model_points = [p for p in model_points if p.jsonl_path in x_mean]
	if not model_points:
		print(f"[{model_display}] no valid jsonl with code[0]; skipping")
		return

	# Global rank normalization (per model): use log2(rank) so doublings are equally spaced.
	# Colorbar will still *display* rank values (1..2048) as tick labels.
	ranks = sorted({p.rank for p in model_points})
	ntrains = sorted({p.num_train_examples for p in model_points})
	# Keep the scale consistent up to 2048 (as requested), even if a model lacks max rank.
	max_rank_for_cbar = max(2048, max(ranks) if ranks else 2048)
	norm = Normalize(vmin=0.0, vmax=math.log2(max_rank_for_cbar))
	cmap = plt.cm.viridis
	size_map = _sizes_from_ntrain(ntrains)

	_apply_style()
	fig, axes = plt.subplots(
		nrows=2,
		ncols=3,
		figsize=(13.5, 6.2),
		sharex=False,
		sharey=True,
	)

	methods = ["avg", "pca"]
	for r, method in enumerate(methods):
		for c, dataset in enumerate(DATASETS):
			ax = axes[r, c]
			sub = [
				p for p in model_points if p.method == method and p.dataset == dataset
			]
			if not sub:
				ax.text(
					0.5,
					0.5,
					"no data",
					ha="center",
					va="center",
					transform=ax.transAxes,
				)
				ax.set_axisbelow(True)
				continue

			x = np.array([x_mean[p.jsonl_path] for p in sub], dtype=float)
			y = np.array([p.acc for p in sub], dtype=float)
			col = np.array(
				[math.log2(p.rank) if p.rank > 0 else 0.0 for p in sub], dtype=float
			)
			sz = np.array(
				[size_map.get(p.num_train_examples, 90.0) for p in sub], dtype=float
			)

			ax.scatter(
				x,
				y,
				c=col,
				s=sz,
				cmap=cmap,
				norm=norm,
				alpha=0.6,
				edgecolors="black",
				linewidths=0.25,
			)

			if r == 0:
				ax.set_title(dataset.upper())
			if c == 0:
				ax.set_ylabel("Accuracy")
				ax.text(
					-0.18,
					0.5,
					method.upper(),
					transform=ax.transAxes,
					rotation=90,
					va="center",
					ha="center",
					fontweight="bold",
				)

			ax.set_xlabel("Avg len(code[0]) over all samples")
			ax.grid(True, which="major", axis="both")
			ax.set_axisbelow(True)

	# Colorbar for rank
	sm = ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
	cbar.set_label("Rank")
	# Show rank values (powers of two) up to 2048 on the colorbar (even though internal scale is log2).
	rank_ticks = [2**k for k in range(0, 12)]  # 1..2048
	ticks_log = [math.log2(r) for r in rank_ticks if math.log2(r) <= norm.vmax]
	cbar.set_ticks(ticks_log)
	cbar.set_ticklabels([str(int(2**t)) for t in ticks_log])

	# Size legend for num_train_examples (show ALL unique values)
	legend_ntrains = sorted(set(ntrains))
	handles = []
	labels = []
	for n in legend_ntrains:
		handles.append(
			plt.scatter(
				[],
				[],
				s=size_map.get(n, 90.0),
				c="#777777",
				alpha=0.7,
				edgecolors="black",
				linewidths=0.35,
			)
		)
		labels.append(str(n))
	if handles:
		# Put legend outside the plotting area (right side) so it doesn't overlap points.
		fig.legend(
			handles,
			labels,
			title="num_train_examples",
			loc="center left",
			bbox_to_anchor=(1.02, 0.18),
			ncol=1,
			framealpha=0.95,
		)

	fig.suptitle(
		f"{model_display}: Accuracy vs Avg Generation Length", y=1.08, fontweight="bold"
	)
	# Leave space on the right for colorbar + num_train_examples legend.
	# (constrained_layout handles colorbar/legend better than tight_layout here)
	fig.set_constrained_layout(True)

	out_path = (
		Path("/home/rishbb/Qwen2.5-Math")
		/ f"acc_vs_len_{_sanitize_filename(model_display)}.png"
	)
	fig.savefig(out_path, bbox_inches="tight")
	print(f"[{model_display}] saved: {out_path}")
	plt.close(fig)


def main():
	print("Scanning transfer results...")
	points = scan_all_points()
	print(f"Found {len(points)} configs total")

	for model_base in TARGET_BASE_MODELS.keys():
		plot_for_model(model_base, points)


if __name__ == "__main__":
	main()
