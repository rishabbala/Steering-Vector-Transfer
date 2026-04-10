#!/usr/bin/env python3
"""
Standalone plot: relative (within-bin) fractions of Correct vs Incorrect by code length.

For each model (rows) and dataset (cols), plot Direct / CoT / Steered, each split into:
  - <THRESH
  - >=THRESH

Each bar is a 100% stack showing correct vs incorrect *within that length bin*.

NOTE (paired filtering):
This script can optionally restrict counts to only those questions that have valid
`code[0]` in BOTH the Direct baseline and the Steered run, and then apply an
additional pairwise length filter between the two generations.

Data sources:
- Steered best config per (model,dataset): max acc across
    /projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_avg
    /projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_pca
- Base Direct/CoT: /projects/llms-lab/transfer_compare/base_evals_test/<model>/<dataset>/*_prompt_general-(direct|cot).jsonl
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt

BASE_DIR = Path("/projects/llms-lab/transfer_compare")
AVG_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_avg"
PCA_DIR = BASE_DIR / "hs_svd_arch_test_transfer_cot_pca"
BASE_EVAL_DIR = BASE_DIR / "base_evals_test"

DATASETS = ["gsm8k", "math", "svamp"]

# Models (base model name -> display label)
TARGET_BASE_MODELS: Dict[str, str] = {
	# "Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": "Qwen-1.5-14B",
	# "gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": "Gemma-2-9B",
	# "OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": "OLMo-2-7B",
	"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": "Qwen-1.5-7B",
	"gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": "Gemma-2-2B",
	"OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": "OLMo-2-1B",
}
PAPER_NAME = {
	"Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B": rf"Qwen-1.5-7B $\longrightarrow$ Qwen-1.5-14B",
	"gemma-2-9b_+_gemma-2-2b_-_gemma-2-2b": rf"gemma-2-2B $\longrightarrow$ gemma-2-9B",
	"OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B": rf"OLMo-2-1B $\longrightarrow$ OLMo-2-7B",
	"Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B": rf"Qwen-1.5-14B $\longrightarrow$ Qwen-1.5-7B",
	"gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b": rf"gemma-2-9B $\longrightarrow$ gemma-2-2B",
	"OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B": rf"OLMo-2-7B $\longrightarrow$ OLMo-2-1B",
}

THRESH = 50


@dataclass(frozen=True)
class BestConfig:
	method: str  # "avg" | "pca"
	base_model: str
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
	d = BASE_EVAL_DIR / base_model / dataset
	if not d.exists():
		return None
	pattern = f"*_prompt_general-{prompt}.jsonl"
	cands = sorted(d.glob(pattern))
	if cands:
		return cands[0]
	# fallback
	cands = sorted(
		p for p in d.glob("*.jsonl") if "metrics" not in p.name and prompt in p.name
	)
	return cands[0] if cands else None


def _scan_best_configs(root: Path, method: str) -> Dict[Tuple[str, str], BestConfig]:
	best: Dict[Tuple[str, str], BestConfig] = {}
	if not root.exists():
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
		# base_model = model_family.split("_+_")[0]
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


def find_best_steered() -> Dict[Tuple[str, str], BestConfig]:
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


def bin_counts(jsonl_path: Path, threshold: int) -> Optional[Dict[str, int]]:
	"""
	Counts only rows where code[0] exists (same convention as earlier scripts).
	Returns raw counts for each bin/correctness.
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

	return {
		"short_total": short_total,
		"short_correct": short_correct,
		"long_total": long_total,
		"long_correct": long_correct,
	}


def _row_key(row: dict) -> Optional[object]:
	"""
	Best-effort stable key for aligning examples across JSONLs.
	Prefers integer-like `idx` used throughout this repo; falls back to question text.
	"""
	if not isinstance(row, dict):
		return None

	k = row.get("idx")
	# Some pipelines may serialize idx as a string.
	if isinstance(k, int):
		return k
	if isinstance(k, str):
		try:
			return int(k)
		except Exception:
			pass

	# Additional common fallbacks (keep conservative)
	for alt in ("example_id", "question_id", "qid", "id"):
		k2 = row.get(alt)
		if isinstance(k2, int):
			return (alt, k2)
		if isinstance(k2, str) and k2:
			return (alt, k2)

	q = row.get("question")
	if isinstance(q, str) and q.strip():
		return ("question", q.strip())
	return None


def _extract_len_and_correct(row: dict) -> Optional[Tuple[int, bool]]:
	code = row.get("code")
	if not isinstance(code, list) or not code:
		return None
	code0 = code[0]
	if not isinstance(code0, str):
		return None
	L = len(code0)

	score = row.get("score")
	is_correct = False
	if isinstance(score, list) and score:
		is_correct = bool(score[0])
	return L, is_correct


def _load_len_correct_by_key(jsonl_path: Path) -> Dict[object, Tuple[int, bool]]:
	"""
	Returns: key -> (len(code[0]), is_correct)
	Only includes rows with valid key and valid code[0].
	"""
	out: Dict[object, Tuple[int, bool]] = {}
	with jsonl_path.open("r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				continue
			if not isinstance(row, dict):
				continue
			k = _row_key(row)
			if k is None:
				continue
			val = _extract_len_and_correct(row)
			if val is None:
				continue
			# First one wins (avoid issues with duplicates)
			out.setdefault(k, val)
	return out


def paired_short_common_stats(
	direct_jsonl_path: Path,
	steered_jsonl_path: Path,
	threshold: int,
) -> Optional[Tuple[int, int, int]]:
	"""
	Compute stats over the *common* questions that are short in BOTH runs:
	1) direct_short_keys = {k : len_direct(k) < threshold}
	2) steered_short_keys = {k : len_steered(k) < threshold}
	3) common_keys = direct_short_keys ∩ steered_short_keys

	Returns:
	  (n_common, direct_correct, steered_correct)
	"""
	try:
		d = _load_len_correct_by_key(direct_jsonl_path)
		s = _load_len_correct_by_key(steered_jsonl_path)
	except FileNotFoundError:
		return None
	except Exception:
		return None

	direct_short_keys = {k for k, (L, _c) in d.items() if L < threshold}
	steered_short_keys = {k for k, (L, _c) in s.items() if L < threshold}
	keys = direct_short_keys & steered_short_keys
	if not keys:
		return None

	n_common = 0
	direct_correct = 0
	steered_correct = 0
	for k in keys:
		dL, dC = d[k]
		sL, sC = s[k]
		# keys already guarantee dL<threshold and sL<threshold; keep defensive check.
		if dL >= threshold or sL >= threshold:
			continue
		n_common += 1
		if dC:
			direct_correct += 1
		if sC:
			steered_correct += 1

	if n_common == 0:
		return None
	return n_common, direct_correct, steered_correct


def _frac(correct: int, total: int) -> Tuple[float, float]:
	"""Return (correct_pct, incorrect_pct) within a bin."""
	if total <= 0:
		return 0.0, 0.0
	c = 100.0 * correct / total
	return c, 100.0 - c


def plot_relative_fraction(best_steered: Dict[Tuple[str, str], BestConfig]) -> Path:
	plt.rcParams.update(
		{
			"figure.dpi": 200,
			"savefig.dpi": 200,
			"axes.titlesize": 12,
			"axes.titleweight": "bold",
			"axes.labelsize": 12,
			"axes.labelweight": "bold",
			"xtick.labelsize": 10,
			"ytick.labelsize": 10,
			"legend.fontsize": 8,
			"hatch.linewidth": 0.8,
		}
	)

	models = list(TARGET_BASE_MODELS.keys())
	datasets = DATASETS

	# One subplot per model; each subplot contains all datasets.
	fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(11, 4), sharey=True)
	if len(models) == 1:
		axes = [axes]

	# Vivid colors (no shading)
	c_correct = "#1f77b4"  # blue
	c_incorrect = "#d62728"  # red

	for i, base_model in enumerate(models):
		ax = axes[i]
		model_label = TARGET_BASE_MODELS[base_model]

		# Horizontal bars: one row per (dataset, method)
		# Each row is a 100%-stack (correct + incorrect) within the <THRESH subset only.
		# Direct vs Steered (paired filtering uses both JSONLs).
		methods = ["Locked", "Unlocked"]

		rows = []
		ylabels = []
		separator_ys = []

		y = 0
		for ds_i, ds in enumerate(datasets):
			direct_path = _find_baseline_jsonl(base_model.split("_+_")[0], ds, "direct")
			steered_cfg = best_steered.get((base_model, ds))
			steered_path = steered_cfg.jsonl_path if steered_cfg else None

			# IMPORTANT: all examples counted must be short (<THRESH) in BOTH Direct and Steered.
			# Therefore we only compute counts when BOTH JSONLs are present; otherwise treat as no data.
			stats = (
				paired_short_common_stats(direct_path, steered_path, THRESH)
				if (direct_path and steered_path)
				else None
			)
			if stats is None:
				for m in methods:
					rows.append((y, 0.0, 0.0))
					ylabels.append(f"{ds.upper()} {m} (no data)")
					y += 1
			else:
				n_common, direct_correct, steered_correct = stats
				print(
					f"{model_label} / {ds}: n_used (short in both, common) = {n_common} | "
					f"Direct correct={direct_correct} | Steered correct={steered_correct}"
				)
				per_method_correct = {
					"Locked": direct_correct,
					"Unlocked": steered_correct,
				}
				for m in methods:
					sc = per_method_correct[m]
					st = n_common
					short_correct, short_incorrect = _frac(sc, st)
					rows.append((y, short_correct, short_incorrect))
					ylabels.append(f"{ds.upper()} {m}")
					y += 1

			separator_ys.append(y - 0.5)

		# Plot rows
		bar_h = 0.72
		for yy, pc, pi in rows:
			ax.barh(
				yy,
				pc,
				height=bar_h,
				color=c_correct,
				label="Correct" if (i == 0 and yy == 0) else None,
			)
			# Percent label inside the correct segment
			if pc >= 2.0:
				ax.text(
					pc / 2,
					yy,
					f"{pc:.0f}%",
					ha="center",
					va="center",
					fontsize=10,
					fontweight="normal",
					color="white",
				)
			ax.barh(
				yy,
				pi,
				left=pc,
				height=bar_h,
				color=c_incorrect,
				label="Incorrect" if (i == 0 and yy == 0) else None,
			)
			# Percent label inside the incorrect segment
			if pi >= 2.0:
				ax.text(
					pc + pi / 2,
					yy,
					f"{pi:.0f}%",
					ha="center",
					va="center",
					fontsize=10,
					fontweight="normal",
					color="white",
				)

		# Cosmetic: separators between datasets
		for sy in separator_ys[:-1]:
			ax.axhline(sy, color="#DDDDDD", linewidth=1.0)

		ax.set_xlim(0, 100)
		ax.set_yticks(range(len(ylabels)))
		ax.set_yticklabels(ylabels, fontsize=8)
		ax.invert_yaxis()
		ax.grid(axis="x")
		ax.set_axisbelow(True)
		ax.set_title(PAPER_NAME[base_model])

		if i == 0:
			ax.set_ylabel("Dataset / method")
		ax.set_xlabel("% within bin")

		ax.margins(x=0.0, y=0.02)

	fig.legend(loc="upper right", bbox_to_anchor=(1, 0.8), framealpha=0.95)

	plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)

	out_path = Path("figures/no_cot_flips_large2small.png")
	fig.savefig(out_path)
	plt.close(fig)
	return out_path


def main():
	best = find_best_steered()
	out = plot_relative_fraction(best)
	print(f"Saved: {out}")


if __name__ == "__main__":
	main()
