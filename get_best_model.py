import json
import os
import re
import sys

try:
	from tabulate import tabulate  # type: ignore
except Exception:
	tabulate = None


def _truncate(s, max_len):
	if s is None:
		return "--"
	s = str(s)
	if max_len is None or max_len <= 0:
		return s
	if len(s) <= max_len:
		return s
	# keep head+tail for filenames/identifiers
	if max_len <= 3:
		return s[:max_len]
	head = max_len - 1
	return s[:head] + "…"


def _print_table(headers, rows):
	"""
	Print a table. Prefer `tabulate` if available; otherwise fall back to a simple formatter.
	"""
	if tabulate is not None:
		# Ensure stable alignment by preventing terminal wrapping.
		# (Long fields like model/metrics_file are truncated upstream.)
		colalign = ["center"] * len(headers)
		print(
			tabulate(
				rows,
				headers=headers,
				tablefmt="github",
				stralign="center",
				numalign="center",
				disable_numparse=True,
				colalign=colalign,
			)
		)
		return

	print(
		"WARNING: `tabulate` is not installed; using a simple formatter.\n"
		"To install: `python -m pip install --user tabulate`"
	)
	widths = [len(str(h)) for h in headers]
	for row in rows:
		for i, cell in enumerate(row):
			widths[i] = max(widths[i], len(str(cell)))

	def fmt_row(r):
		return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))

	sep = "-+-".join("-" * w for w in widths)
	print(fmt_row(headers))
	print(sep)
	for r in rows:
		print(fmt_row(r))


"""
Find best-performing model configuration per dataset within test result folders.

For a given `root_pth`, this script scans for `*_metrics.json` files and picks the
single configuration (model_family + num_train_samples + rank + alpha) with maximum
`acc` for each dataset.
"""

# Default folders (override by passing CLI args)
DEFAULT_AVG_ROOT = (
	# None
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_avg/"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_avg_icl_direct,cot+icl"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_avg_icl_cot,cot+icl"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_cot_avg_grid_search/"
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_ID_avg"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_avg"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_test_cot_avg_model_fam"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_test_multilingual_transfer_avg"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_avg_model_fam"
	# "/projects/llms-lab/transfer_compare/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_icl_avg_direct,cot+icl"
)
DEFAULT_PCA_ROOT = (
	# None
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_pca/"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_pca_icl_direct,cot+icl"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_pca_icl_cot,cot+icl"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_cot_pca_grid_search/"
	"/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_ID_pca"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_test_cot_pca_model_fam"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_transfer_test_multilingual_transfer_pca"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_cot_pca"
	# "/projects/llms-lab/transfer_compare/hs_svd_arch_test_transfer_post_training_OOD_pca_model_fam"
)

# If you don't pass model names on the CLI, these are used.
# Provide model names like: "Qwen1.5-14B" or "gemma-2-9b" etc.
DEFAULT_MODELS = [
	# "Qwen1.5-7B_+_Qwen1.5-14B_-_Qwen1.5-14B"
	# "Qwen1.5-14B_+_Qwen1.5-7B_-_Qwen1.5-7B"
	# "OLMo-2-1124-13B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B"
	# "OLMo-2-1124-7B_+_OLMo-2-1124-13B_-_OLMo-2-1124-13B"
	# "OLMo-2-1124-7B_+_OLMo-2-0425-1B_-_OLMo-2-0425-1B"
	# "OLMo-2-0425-1B_+_OLMo-2-1124-7B_-_OLMo-2-1124-7B"
	# "gemma-2-2b_+_gemma-2-9b_-_gemma-2-9b"
	# "Qwen2.5-1.5B_+_Qwen2.5-7B-Instruct_-_Qwen2.5-7B"
	# "Qwen2.5-7B_+_Qwen2.5-1.5B-Instruct_-_Qwen2.5-1.5B"
	# "Qwen2.5-1.5B_+_Qwen2.5-7B-Instruct_-_Qwen2.5-7B",
	# "Qwen3-14B-Base_+_Qwen3-4B_-_Qwen3-4B-Base",
	# "Qwen3-8B-Base_+_Qwen3-4B_-_Qwen3-4B-Base",
	# "Qwen3-4B-Base_+_Qwen3-14B_-_Qwen3-14B-Base",
	# "Qwen3-4B-Base_+_Qwen3-8B_-_Qwen3-8B-Base",
	# "Qwen2.5-7B_+_OpenReasoning-Nemotron-14B_-_Qwen2.5-14B"
	# "Qwen2.5-1.5B_+_DLER-R1-7B-Research_-_Qwen2.5-7B"
	# "gemma-3-12b-pt_+_gemma-3-4b-it_-_gemma-3-4b-pt"
	# "Qwen3-4B-Base_+_Nemotron-Cascade-8B_-_Qwen3-8B-Base"
	# "Qwen2.5-14B_+_DLER-R1-7B-Research_-_Qwen2.5-7B"
	# "Qwen3-8B-Base_+_Qwen3-4B-Thinking-2507_-_Qwen3-4B-Base"
	# "Qwen3-14B-Base_+_DeepSeek-R1-0528-Qwen3-8B_-_Qwen3-8B-Base"
	# "gemma-2-9b_+_Qwen1.5-7B-Chat_-_Qwen1.5-7B"
	# "Ministral-3-3B-Base-2512_+_Ministral-3-8B-Instruct-2512-BF16_-_Ministral-3-8B-Base-2512"
	# "Ministral-3-3B-Base-2512_+_Ministral-3-14B-Instruct-2512-BF16_-_Ministral-3-14B-Base-2512"
	"Ministral-3-8B-Base-2512_+_Ministral-3-3B-Instruct-2512-BF16_-_Ministral-3-3B-Base-2512"
	# "Ministral-3-14B-Base-2512_+_Ministral-3-3B-Instruct-2512-BF16_-_Ministral-3-3B-Base-2512"
]


# ----------------------------
# Utility functions
# ----------------------------
def collectDatasetDirs(root_dir):
	if not root_dir or not os.path.isdir(root_dir):
		return []
	return sorted(
		[
			os.path.join(root_dir, d)
			for d in os.listdir(root_dir)
			if os.path.isdir(os.path.join(root_dir, d))
		]
	)


def collectRankDirs(root_dir):
	if not root_dir or not os.path.isdir(root_dir):
		return []
	rank_dirs = []
	for d in os.listdir(root_dir):
		full = os.path.join(root_dir, d)
		if os.path.isdir(full) and re.fullmatch(r"rank_\d+", d):
			rank_dirs.append(full)
	return sorted(rank_dirs)


def parseRankFromDir(rank_dir):
	m = re.search(r"rank_(\d+)", os.path.basename(rank_dir.rstrip("/")))
	return int(m.group(1)) if m else None


def parseNumTrainSamplesFromPath(path):
	# Match any segment like num_train_samples_128
	m = re.search(r"(?:^|/)num_train_samples_(\d+)(?:/|$)", path)
	return int(m.group(1)) if m else None


def parseModelFamilyFromPath(path):
	"""
	Assumes the folder structure:
	  .../num_train_samples_<N>/<model_family>/rank_<R>/...
	Returns model_family if found.
	"""
	m = re.search(r"(?:^|/)num_train_samples_\d+/([^/]+)/rank_\d+(?:/|$)", path)
	return m.group(1) if m else None


def parseRankFromPath(path):
	m = re.search(r"(?:^|/)rank_(\d+)(?:/|$)", path)
	return int(m.group(1)) if m else None


def parseAlphaFromFilename(filename):
	m = re.search(r"alpha_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", filename)
	return float(m.group(1)) if m else None


def readAccFromJson(path):
	with open(path, "r") as f:
		data = json.load(f)
	return data.get("acc", None)


def bestConfigPerDataset(root_pth):
	"""
	Scan all *_metrics.json files under root_pth and return:
	  {dataset: best_info_dict}
	where best_info_dict includes model_family, num_train_samples, rank, alpha, acc, metrics_file.
	"""
	results = {}
	for subdir, _, files in os.walk(root_pth):
		for f in files:
			if not (
				f.endswith("_metrics.json") or (f.endswith(".json") and "metrics" in f)
			):
				continue
			metrics_path = os.path.join(subdir, f)
			dataset = os.path.basename(os.path.dirname(metrics_path))
			try:
				acc = readAccFromJson(metrics_path)
			except Exception:
				continue
			if acc is None:
				continue

			model_family = parseModelFamilyFromPath(metrics_path)
			num_train_samples = parseNumTrainSamplesFromPath(metrics_path)
			rank = parseRankFromPath(metrics_path)
			alpha = parseAlphaFromFilename(f)

			info = {
				"acc": float(acc),
				"alpha": alpha,
				"rank": rank,
				"num_train_samples": num_train_samples,
				"model_family": model_family,
				"metrics_file": f,
				"metrics_path": metrics_path,
			}
			prev = results.get(dataset)
			if prev is None or info["acc"] > prev["acc"]:
				results[dataset] = info
	return results


def _model_matches(model_family: str, model_query: str) -> bool:
	"""
	Return True if a metrics file's model_family matches the user-provided model name.
	We match either:
	- base model name (prefix before '_+_') equals model_query
	- OR model_query appears as a substring (case-sensitive) in model_family
	"""
	if not model_family:
		return False
	base = model_family.split("_+_")[0]
	return base == model_query or (model_query in model_family)


def bestConfigPerDatasetPerModel(root_pth: str, model_queries) -> dict:
	"""
	Returns:
	  {(dataset, model_query): best_info_dict}
	where best is chosen by max acc among configs that match that model_query.
	"""
	results = {}
	model_queries = [m for m in model_queries if m]
	for subdir, _, files in os.walk(root_pth):
		for f in files:
			if not (
				f.endswith("_metrics.json") or (f.endswith(".json") and "metrics" in f)
			):
				continue
			metrics_path = os.path.join(subdir, f)
			dataset = os.path.basename(os.path.dirname(metrics_path))
			# if any(x in dataset for x in ["mmlu", "arc_c"]):
			# 	continue
			try:
				acc = readAccFromJson(metrics_path)
			except Exception:
				continue
			if acc is None:
				continue

			model_family = parseModelFamilyFromPath(metrics_path)
			if not model_family:
				continue

			for mq in model_queries:
				if not _model_matches(model_family, mq):
					continue

				num_train_samples = parseNumTrainSamplesFromPath(metrics_path)
				rank = parseRankFromPath(metrics_path)
				alpha = parseAlphaFromFilename(f)
				info = {
					"acc": float(acc),
					"alpha": alpha,
					"rank": rank,
					"num_train_samples": num_train_samples,
					"model_family": model_family,
					"metrics_file": f,
					"metrics_path": metrics_path,
					"model_query": mq,
				}
				key = (dataset, mq)
				prev = results.get(key)
				if prev is None or info["acc"] > prev["acc"]:
					results[key] = info
	return results


def main():
	# Mode A (compare): avg_path pca_path
	# Usage: python get_best_model.py /path/to/avg /path/to/pca
	# Mode B (per-model): root_path ModelA [ModelB ...]
	# Usage: python get_best_model.py /path/to/root ModelA [ModelB ...]

	argv = sys.argv[1:]

	def _as_optional_dir(s):
		# Allow passing 'none' to disable one side.
		if s is None:
			return None
		if str(s).lower() in {"none", "null", "--"}:
			return None
		return s

	# Compare mode if at least two args are provided:
	#   python get_best_model.py AVG_DIR PCA_DIR [ModelA ModelB ...]
	# either AVG_DIR or PCA_DIR can be 'none'
	if len(argv) >= 2:
		avg_root = _as_optional_dir(argv[0])
		pca_root = _as_optional_dir(argv[1])
		model_queries = argv[2:] if len(argv) > 2 else DEFAULT_MODELS
	elif len(argv) == 0:
		avg_root, pca_root = DEFAULT_AVG_ROOT, DEFAULT_PCA_ROOT
		model_queries = DEFAULT_MODELS
	else:
		avg_root = None
		pca_root = None

	# Compare (avg/pca) mode if at least one of the roots is provided.
	if avg_root is not None or pca_root is not None:
		# Validate any provided dirs
		if avg_root is not None and not os.path.isdir(avg_root):
			print(f"ERROR: avg folder does not exist: {avg_root}")
			sys.exit(2)
		if pca_root is not None and not os.path.isdir(pca_root):
			print(f"ERROR: pca folder does not exist: {pca_root}")
			sys.exit(2)

		model_queries = [m for m in (model_queries or []) if m]
		if not model_queries:
			print(
				"ERROR: No model names provided for compare mode.\n"
				"Set DEFAULT_MODELS or pass model names after the two folder paths:\n"
				"  python get_best_model.py AVG_DIR PCA_DIR ModelA [ModelB ...]"
			)
			sys.exit(2)

		# Best per (dataset, model_query) inside each folder, then choose avg vs pca.
		avg_best = (
			bestConfigPerDatasetPerModel(avg_root, model_queries) if avg_root else {}
		)
		pca_best = (
			bestConfigPerDatasetPerModel(pca_root, model_queries) if pca_root else {}
		)

		datasets = sorted(
			set(ds for (ds, _) in avg_best.keys())
			| set(ds for (ds, _) in pca_best.keys())
		)

		# If multiple model queries, include the query column to disambiguate rows.
		multi = len(model_queries) > 1
		headers = ["Dataset"]
		if multi:
			headers.append("ModelQuery")
		headers += ["Best", "Acc", "Model", "metrics_path"]

		rows = []
		for ds in datasets:
			for mq in model_queries:
				a = avg_best.get((ds, mq))
				p = pca_best.get((ds, mq))
				if a is None and p is None:
					continue
				if p is None:
					best_label = "avg"
					info = a
				elif a is None:
					best_label = "pca"
					info = p
				elif a["acc"] >= p["acc"]:
					best_label = "avg"
					info = a
				else:
					best_label = "pca"
					info = p

				row = [ds]
				if multi:
					row.append(mq)
				row += [
					best_label,
					f"{info['acc']:.1f}",
					info.get("model_family") or "--",
					info.get("metrics_path") or "--",
				]
				rows.append(row)

		_print_table(headers, rows)
		return

	# Per-model mode
	root = argv[0] if len(argv) > 0 else DEFAULT_AVG_ROOT
	if not os.path.isdir(root):
		print(f"ERROR: folder does not exist: {root}")
		print("Usage: python get_best_model.py /path/to/avg /path/to/pca")
		print("   or: python get_best_model.py /path/to/root ModelA [ModelB ...]")
		sys.exit(2)

	model_queries = argv[1:] if len(argv) > 1 else DEFAULT_MODELS
	if not model_queries:
		print(
			"ERROR: No model names provided.\n"
			"Usage: python get_best_model.py /path/to/root ModelA [ModelB ...]"
		)
		sys.exit(2)

	best_map = bestConfigPerDatasetPerModel(root, model_queries)
	datasets = sorted({ds for (ds, _) in best_map.keys()})

	headers = ["Dataset", "Acc", "Model", "metrics_path"]
	rows = []
	for ds in datasets:
		for mq in model_queries:
			info = best_map.get((ds, mq))
			if info is None:
				continue
			rows.append(
				[
					ds,
					f"{info['acc']:.1f}",
					info.get("model_family") or "--",
					info.get("metrics_path") or "--",
				]
			)

	_print_table(headers, rows)


if __name__ == "__main__":
	main()
