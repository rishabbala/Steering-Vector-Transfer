import asyncio
import json
import os
from pathlib import Path

import google.generativeai as genai
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Configure API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Batch settings
BATCH_SIZE = 10  # Number of concurrent API calls
MAX_RETRIES = 100
RETRY_DELAY = 10


async def checkAnswerAsync(prompt, gt, code_field, semaphore):
	"""Async version of checkAnswer with rate limiting"""
	eval_prompt = f"""You are an expert evaluator who will check the solutions generated. You are given the input question along with any options for MCQs, the ground truth solution, and the generated answer. Check if the generated answer contains the grount truth solution.
	Input prompt: {prompt}
	Ground truth solution: {gt}
	Generated answer: {code_field}
	Does the generated answer contain the ground truth solution "{gt}" or any equivalent form of the solution? 
	- For multiple choice questions, check if the correct answer or choice is selected as the final answer in the generated final answer.
	- For questions with numerical answers, check if the answer is present in the text. Look for the exact numerical value or very close approximations upto two decimal places.
	- For answers with equations, check if the two equations are equivalent when simplified.
	- If the model performs multiple rounds of reasoning, check if the final answer is present in the last round.
	- Only check the answer to the question provided, and not any other questions that the model may have generated. If  the model repeats the provided question, check the answer generated after the repetition.
	- Ignore the sentences after phrases like "The final answer is" or "</atok>" or similar concluding statements. Only check the answer to the question provided.
	- Answer with only "YES" or "NO"
	"""

	async with semaphore:
		for attempt in range(MAX_RETRIES):
			try:
				# Create a new model instance for each call to avoid conflicts
				model = genai.GenerativeModel("gemini-2.5-flash-lite")
				response = await asyncio.to_thread(model.generate_content, eval_prompt)
				return (
					response.text.strip().upper() == "YES"
					or "yes" in response.text.strip().lower()
				)
			except Exception as e:
				if attempt < MAX_RETRIES - 1:
					await asyncio.sleep(RETRY_DELAY)
				else:
					print(f"Max retries reached for gt={gt}: {e}")
					return False

	return False


async def processBatch(batch_data, semaphore):
	"""Process a batch of items asynchronously"""
	tasks = []
	for prompt, gt, code in batch_data:
		task = checkAnswerAsync(prompt, gt, code, semaphore)
		tasks.append(task)

	results = await asyncio.gather(*tasks)
	return results


async def processJsonlFileAsync(jsonl_path):
	"""Async process a single JSONL file and return accuracy metrics"""
	correct_cnt = 0
	total_cnt = 0

	print(f"\nProcessing: {jsonl_path}")

	with open(jsonl_path, "r") as f:
		lines = f.readlines()

	# Prepare all data
	batch_data = []
	for line in lines:
		try:
			data = json.loads(line)
			prompt = data["prompt"]
			gt = data["gt"]
			code = data["code"][0] if isinstance(data["code"], list) else data["code"]
			batch_data.append((prompt, gt, code))
		except Exception as e:
			print(f"Error parsing line: {e}")
			continue

	total_cnt = len(batch_data)

	# Create semaphore for rate limiting
	semaphore = asyncio.Semaphore(BATCH_SIZE)

	# Process in batches with progress bar
	all_results = []
	for i in tqdm(range(0, len(batch_data), BATCH_SIZE), desc="Processing batches"):
		batch = batch_data[i : i + BATCH_SIZE]
		results = await processBatch(batch, semaphore)
		all_results.extend(results)

	correct_cnt = sum(all_results)
	acc = (correct_cnt / total_cnt * 100) if total_cnt > 0 else 0.0

	# Create detailed results with evaluations
	detailed_results = []
	for (prompt, gt, code), is_correct in zip(batch_data, all_results):
		detailed_results.append(
			{"prompt": prompt, "gt": gt, "code": code, "gemini_correct": is_correct}
		)

	return {"num_samples": total_cnt, "acc": round(acc, 1)}, detailed_results


def collectAllFiles(base_dirs):
	"""Collect all JSONL files grouped by dataset name"""
	dataset_files = {}

	for base_dir in base_dirs:
		base_path = Path(base_dir)
		jsonl_files = list(base_path.rglob("*.jsonl"))

		for jsonl_file in jsonl_files:
			# Skip if metrics already exist or if mmlu or not model_name
			metrics_file = jsonl_file.parent / f"{jsonl_file.stem}_metrics-gemini.json"

			print("Base path: ", str(base_path))
			print("Jsonl file: ", str(jsonl_file))

			if (
				metrics_file.exists()
				or "mmlu" in str(jsonl_file).lower()
				or "math500" not in str(jsonl_file).lower()
				or "qwen3" not in str(jsonl_file).lower()
			):
				print(
					f"Skipping {jsonl_file.stem} because it is mmlu or metrics already exist"
				)
				continue

			# Extract dataset name (parent folder name)
			dataset_name = jsonl_file.parent.name

			if dataset_name not in dataset_files:
				dataset_files[dataset_name] = []

			dataset_files[dataset_name].append(jsonl_file)

	return dataset_files


async def processAllFilesByDataset(base_dirs):
	"""Process all JSONL files grouped by dataset"""
	dataset_files = collectAllFiles(base_dirs)

	print(f"Found {len(dataset_files)} unique datasets")
	for dataset_name, files in dataset_files.items():
		print(f"  {dataset_name}: {len(files)} files")

	# Process dataset by dataset
	for dataset_name, jsonl_files in dataset_files.items():
		print(f"\n{'=' * 80}")
		print(f"Processing dataset: {dataset_name}")
		print(f"{'=' * 80}")

		for jsonl_file in jsonl_files:
			metrics_file = jsonl_file.parent / f"{jsonl_file.stem}_metrics-gemini.json"
			detailed_file = jsonl_file.parent / f"{jsonl_file.stem}-gemini.jsonl"

			# Process the file
			metrics, detailed_results = await processJsonlFileAsync(jsonl_file)

			# Save metrics
			with open(metrics_file, "w") as f:
				json.dump(metrics, f, indent=4)

			# Save detailed results
			with open(detailed_file, "w") as f:
				for result in detailed_results:
					f.write(json.dumps(result) + "\n")

			print(f"Saved metrics to: {metrics_file}")
			print(f"Saved detailed results to: {detailed_file}")
			print(f"Results: {metrics}")


if __name__ == "__main__":
	# for iteration in range(2):
	print(f"\n{'#' * 80}")
	# print(f"Starting iteration {iteration + 1}/2")
	print(f"{'#' * 80}")

	base_dirs = [
		"/projects/llms-lab/pinjie/Steering-Vector-Transfer/outputs/results_avg_ID_post_train",
		"/projects/llms-lab/transfer_compare/base_evals_test/Qwen3-8B-Base",
		"/projects/llms-lab/transfer_compare/base_evals_test/Qwen3-4B-Base",
		"/projects/llms-lab/transfer_compare/base_evals_test/Qwen3-4B-Thinking-2507",
	]

	asyncio.run(processAllFilesByDataset(base_dirs))
