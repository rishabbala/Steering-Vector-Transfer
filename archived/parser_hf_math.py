"""
Parser using HuggingFace Math-Verify package for robust mathematical answer extraction and verification.

This module provides a wrapper around the hf-math-verify package to integrate with the existing
parser.py workflow while leveraging more robust answer extraction and comparison capabilities.

Installation:
	pip install math-verify[antlr4-python3-runtime]

Reference:
	https://github.com/huggingface/Math-Verify

Usage as Drop-in Replacement:
	To use this parser instead of the original parser.py in base_class.py, simply change:
	
		from parser import run_execute, parse_ground_truth, parse_question
	
	to:
	
		from parser_hf_math import run_execute, parse_ground_truth, parse_question
	
	The function signatures are identical, so no other code changes are needed.
	If math-verify is not installed, it automatically falls back to the original parser.

Key Functions (matching parser.py):
	- extract_answer(pred_str, data_name, samples, use_last_number=True)
	- run_execute(executor, result, prompt_type, data_name, execute=False, samples=None)
	- parse_ground_truth(example, data_name)  [imported from parser.py]
	- parse_question(example, data_name)  [imported from parser.py]

Additional Functions:
	- verify_answer_hf(prediction, gold, data_name, ...)
	- parse_and_verify_hf(example, prediction_str, data_name, ...)
	- evaluate_predictions_hf(examples, predictions, data_name, ...)
"""

import re
from typing import Any, Dict, Optional, Tuple

try:
	from math_verify import parse, verify
	from math_verify.parser import (
		LatexExtractionConfig,
		ExprExtractionConfig,
		StringExtractionConfig,
	)
	MATH_VERIFY_AVAILABLE = True
except ImportError:
	MATH_VERIFY_AVAILABLE = False
	print("Warning: math-verify package not installed. Install with: pip install math-verify[antlr4-python3-runtime]")

# Import utilities from the original parser
from parser import (
	parse_ground_truth,
	parse_question,
	extract_answer as fallback_extract_answer,
	STRIP_EXCEPTIONS,
)


def get_extraction_config(data_name: str, is_gold: bool = False):
	"""
	Get appropriate extraction configuration based on dataset.
	
	Args:
		data_name: Name of the dataset
		is_gold: Whether this is for gold answer (True) or prediction (False)
	
	Returns:
		List of extraction configs for math_verify.parse()
	"""
	# Multiple choice questions - use string extraction
	if data_name in ["arc_c", "gpqa", "commonsense_qa", "mmlu_stem", "mmlu_pro", "winogrande"]:
		return [StringExtractionConfig()]
	
	# Boolean questions
	if data_name in ["strategyqa"]:
		return [StringExtractionConfig()]
	
	# Math datasets with LaTeX
	if any(x in data_name for x in [
		"math", "math500", "minerva_math", "olympiadbench", 
		"gaokao", "college_math", "agieval_math", "deepmind_math"
	]):
		if is_gold:
			# Gold answers are typically clean LaTeX
			return [LatexExtractionConfig()]
		else:
			# Predictions might be in various formats
			# Prioritize boxed answers for predictions
			return [
				LatexExtractionConfig(boxed_match_priority=0),
				ExprExtractionConfig()
			]
	
	# Numeric math problems (GSM8K, SVAMP, etc.)
	if data_name in ["gsm8k", "svamp", "aime24", "amc23"] or "mgsm" in data_name:
		return [ExprExtractionConfig()]
	
	# Default: try both LaTeX and expression extraction
	return [LatexExtractionConfig(), ExprExtractionConfig()]


def extract_answer(
	pred_str: str,
	data_name: str,
	samples: Optional[Dict[str, Any]] = None,
	use_last_number: bool = True,
	use_fallback: bool = True
) -> str:
	"""
	Extract answer from prediction string using math-verify package.
	
	This function signature matches parser.py's extract_answer() for compatibility.
	
	Args:
		pred_str: The model's prediction string
		data_name: Name of the dataset
		samples: Sample data (for context like multiple choice options)
		use_last_number: Kept for compatibility with original parser (unused here)
		use_fallback: Whether to fall back to original parser if math-verify fails
	
	Returns:
		Extracted answer string
	"""
	if not MATH_VERIFY_AVAILABLE:
		if use_fallback:
			return fallback_extract_answer(pred_str, data_name, samples, use_last_number)
		return ""
	
	# Clean up common artifacts
	pred_str = pred_str.replace("\u043a\u0438", "")
	
	# Get appropriate extraction config
	extraction_config = get_extraction_config(data_name, is_gold=False)
	
	try:
		# Parse the prediction
		parsed_answer = parse(pred_str, extraction_config=extraction_config)
		
		# If parsing succeeded, return the string representation
		if parsed_answer is not None:
			return str(parsed_answer)
	except Exception as e:
		# If math-verify fails, optionally fall back to original parser
		if use_fallback:
			return fallback_extract_answer(pred_str, data_name, samples, use_last_number)
	
	return ""


def verify_answer_hf(
	prediction: str,
	gold: str,
	data_name: str,
	allow_set_relation_comp: bool = False,
	float_tolerance: float = 1e-4
) -> bool:
	"""
	Verify if prediction matches gold answer using math-verify package.
	
	Args:
		prediction: The extracted prediction
		gold: The gold answer
		data_name: Name of the dataset
		allow_set_relation_comp: Allow bidirectional set/inequality comparison
		float_tolerance: Tolerance for floating point comparisons
	
	Returns:
		True if answers match, False otherwise
	"""
	if not MATH_VERIFY_AVAILABLE:
		# Fall back to string comparison
		return str(prediction).strip() == str(gold).strip()
	
	try:
		# Get extraction configs
		pred_config = get_extraction_config(data_name, is_gold=False)
		gold_config = get_extraction_config(data_name, is_gold=True)
		
		# Parse both answers
		parsed_pred = parse(prediction, extraction_config=pred_config)
		parsed_gold = parse(gold, extraction_config=gold_config)
		
		# Verify
		return verify(
			parsed_gold,
			parsed_pred,
			allow_set_relation_comp=allow_set_relation_comp,
			float_tolerance=float_tolerance
		)
	except Exception as e:
		# Fall back to string comparison
		return str(prediction).strip() == str(gold).strip()


def run_execute(
	executor,
	result: str,
	prompt_type: str,
	data_name: str,
	execute: bool = False,
	samples: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[Any]]:
	"""
	Execute and extract answer using math-verify (compatible with original run_execute).
	
	This function signature matches parser.py's run_execute() for compatibility.
	
	Args:
		executor: Code executor (if needed)
		result: Model output string
		prompt_type: Type of prompt used
		data_name: Dataset name
		execute: Whether to execute code
		samples: Sample data
	
	Returns:
		Tuple of (prediction, report)
	"""
	if not result or result == "error":
		return None, None
	
	report = None
	prediction = extract_answer(result, data_name, samples, use_fallback=True)
	
	return prediction, report


def parse_and_verify_hf(
	example: Dict[str, Any],
	prediction_str: str,
	data_name: str,
	allow_set_relation_comp: bool = False,
	float_tolerance: float = 1e-4
) -> Tuple[str, str, bool]:
	"""
	Complete pipeline: parse ground truth, extract prediction, and verify.
	
	Args:
		example: Example dict containing ground truth
		prediction_str: Model's prediction string
		data_name: Dataset name
		allow_set_relation_comp: Allow bidirectional set/inequality comparison
		float_tolerance: Tolerance for floating point comparisons
	
	Returns:
		Tuple of (gold_answer, predicted_answer, is_correct)
	"""
	# Parse ground truth using original parser
	_, gold_ans = parse_ground_truth(example, data_name)
	
	# Extract prediction using math-verify
	pred_ans = extract_answer(prediction_str, data_name, example)
	
	# Verify using math-verify
	is_correct = verify_answer_hf(
		pred_ans,
		gold_ans,
		data_name,
		allow_set_relation_comp=allow_set_relation_comp,
		float_tolerance=float_tolerance
	)
	
	return gold_ans, pred_ans, is_correct


# Convenience function for batch processing
def evaluate_predictions_hf(
	examples: list,
	predictions: list,
	data_name: str,
	allow_set_relation_comp: bool = False,
	float_tolerance: float = 1e-4,
	verbose: bool = False
) -> Dict[str, Any]:
	"""
	Evaluate a batch of predictions against ground truth.
	
	Args:
		examples: List of example dicts with ground truth
		predictions: List of prediction strings
		data_name: Dataset name
		allow_set_relation_comp: Allow bidirectional set/inequality comparison
		float_tolerance: Tolerance for floating point comparisons
		verbose: Print detailed results
	
	Returns:
		Dict with accuracy and detailed results
	"""
	assert len(examples) == len(predictions), "Mismatch between examples and predictions"
	
	results = []
	correct = 0
	
	for i, (example, pred_str) in enumerate(zip(examples, predictions)):
		gold, pred, is_correct = parse_and_verify_hf(
			example,
			pred_str,
			data_name,
			allow_set_relation_comp=allow_set_relation_comp,
			float_tolerance=float_tolerance
		)
		
		if is_correct:
			correct += 1
		
		result = {
			"idx": i,
			"gold": gold,
			"prediction": pred,
			"correct": is_correct
		}
		results.append(result)
		
		if verbose:
			status = "✓" if is_correct else "✗"
			print(f"{status} [{i}] Gold: {gold} | Pred: {pred}")
	
	accuracy = correct / len(examples) if examples else 0.0
	
	return {
		"accuracy": accuracy,
		"correct": correct,
		"total": len(examples),
		"results": results
	}


# Backward compatibility aliases
extract_answer_hf = extract_answer
run_execute_hf = run_execute


if __name__ == "__main__":
	# Example usage
	if MATH_VERIFY_AVAILABLE:
		# Test basic extraction
		test_pred = "Let me solve this step by step... Therefore, the answer is $\\boxed{166}$ cm²"
		extracted = extract_answer(test_pred, "math", None)
		print(f"Extracted: {extracted}")
		
		# Test verification
		gold = "166"
		is_correct = verify_answer_hf(extracted, gold, "math")
		print(f"Verification: {is_correct}")
		
		# Test with different formats
		test_cases = [
			("$\\frac{1}{2}$", "0.5", "math"),
			("${1,2,3}$", "${1,3,2}$", "math"),
			("A", "A", "mmlu_stem"),
		]
		
		print("\nTest cases:")
		for pred, gold, dataset in test_cases:
			result = verify_answer_hf(pred, gold, dataset)
			print(f"  {pred} == {gold} ({dataset}): {result}")
	else:
		print("math-verify not installed. Please install with:")
		print("  pip install math-verify[antlr4-python3-runtime]")
