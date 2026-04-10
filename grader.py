"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import multiprocessing
import re
from math import isclose
from typing import Union

import regex
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


def choice_answer_clean(pred: str):
	pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
	# Clean the answer based on the dataset
	tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
	if tmp:
		pred = tmp
	else:
		pred = [pred.strip().strip(".")]
	pred = pred[-1]
	# Remove the period at the end, again!
	pred = pred.rstrip(".").rstrip("/")
	return pred


def parse_digits(num):
	num = regex.sub(r"(,|\{,\}|\\!)", "", str(num))
	try:
		return float(num)
	except:
		if num.endswith("%"):
			num = num[:-1]
			if num.endswith("\\"):
				num = num[:-1]
			try:
				return float(num) / 100
			except:
				pass
	return None


def is_digit(num):
	# paired with parse_digits
	return parse_digits(num) is not None


def str_to_pmatrix(input_str):
	input_str = input_str.strip()
	matrix_str = re.findall(r"\{.*,.*\}", input_str)
	pmatrix_list = []

	for m in matrix_str:
		m = m.strip("{}")
		pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
		pmatrix_list.append(pmatrix)

	return ", ".join(pmatrix_list)


def math_equal(
	prediction: Union[bool, float, str],
	reference: Union[float, str],
	include_percentage: bool = True,
	is_close: bool = True,
	timeout: bool = False,
	recursive_timeout: int = 10,
	recursive_count: int = 0,
) -> bool:
	"""
	Exact match of math if and only if:
	1. numerical equal: both can convert to float and are equal
	2. symbolic equal: both can convert to sympy expression and are equal
	"""

	if prediction is None or reference is None:
		return False

	try:
		prediction = prediction.lstrip("[](){}\\/")
		reference = reference.lstrip("[](){}\\/")
	except:
		return False

	if recursive_count > recursive_timeout:
		return False

	if re.search(r"\b[\w\.]+\s*(\/|per)\s*s(\^?\{?\d*\}?)?", prediction, re.IGNORECASE):
		prediction = re.sub(r"\b[\w\.]+\s*(\/|per)\s*s(\^?\{?\d*\}?)?", "", prediction)

	# Replace e^{x}, e^x, \times 10^{x}
	def _replace_exponentials(s):
		s = re.sub(
			r"(\d+(?:\.\d+)?)[eE]([-+]?\d+)",
			lambda m: m.group(1) + r"\\times 10^{" + m.group(2) + r"}",
			s,
		)
		s = re.sub(
			r"(?<![A-Za-z0-9\\])e([-+]?\d+)(?![A-Za-z0-9])",
			lambda m: r"\\times 10^{" + m.group(1) + r"}",
			s,
		)

		return s

	prediction = _replace_exponentials(prediction)
	reference = _replace_exponentials(reference)

	# print("prediction", prediction)
	# print("reference", reference)

	prediction = prediction.replace("np.", "")
	reference = reference.replace("np.", "")
	prediction = prediction.replace("} {", "}{")
	reference = reference.replace("} {", "}{")

	## replace for fracs
	try:
		if "\\frac" in prediction:
			pred_parts = prediction.split("frac")[1]
			pred_a, pred_b = pred_parts.split("}{")[0], pred_parts.split("}{")[1]
			pred_a = pred_a.strip("{}").strip()
			pred_b = pred_b.strip("{}").strip()
			prediction = prediction.replace(
				f"\\frac{{{pred_a}}}{{{pred_b}}}", f"{pred_a}/{pred_b}"
			)
		if "\\frac" in reference:
			ref_parts = reference.split("frac")[1]
			ref_a, ref_b = ref_parts.split("}{")[0], ref_parts.split("}{")[1]
			ref_a = ref_a.strip("{}").strip()
			ref_b = ref_b.strip("{}").strip()
			reference = reference.replace(
				f"\\frac{{{ref_a}}}{{{ref_b}}}", f"{ref_a}/{ref_b}"
			)
	except:
		print("Exception in replacing fracs", prediction, reference)

	# print("Judge:", prediction, reference)
	if prediction is None or reference is None:
		return False
	if str(prediction.strip().lower()) == str(reference.strip().lower()):
		return True
	if reference in ["A", "B", "C", "D", "E", "(A)", "(B)", "(C)", "(D)", "(E)"] and (
		choice_answer_clean(prediction) == reference
		or choice_answer_clean(prediction) == "(" + reference + ")"
		or "(" + choice_answer_clean(prediction) + ")" == reference
	):
		return True

	try:  # 1. numerical equal
		if is_digit(prediction) and is_digit(reference):
			prediction = parse_digits(prediction)
			reference = parse_digits(reference)

			# number questions
			if include_percentage:
				gt_result = [reference / 100, reference, reference * 100]
			else:
				gt_result = [reference]
			for item in gt_result:
				try:
					if is_close:
						if numeric_equal(prediction, item):
							return True
					else:
						if item == prediction:
							return True
				except Exception:
					continue
			return False
	except:
		pass

	if not prediction and prediction not in [0, False]:
		return False

	# 2. symbolic equal
	reference = str(reference).strip()
	prediction = str(prediction).strip()

	## pmatrix (amps)
	if "pmatrix" in prediction and "pmatrix" not in reference:
		reference = str_to_pmatrix(reference)

	## deal with [], (), {}
	pred_str, ref_str = prediction, reference
	if (
		prediction.startswith("[")
		and prediction.endswith("]")
		and not reference.startswith("(")
	) or (
		prediction.startswith("(")
		and prediction.endswith(")")
		and not reference.startswith("[")
	):
		pred_str = pred_str.strip("[]()")
		ref_str = ref_str.strip("[]()")
	for s in ["{", "}", "(", ")"]:
		ref_str = ref_str.replace(s, "")
		pred_str = pred_str.replace(s, "")
	if pred_str.lower() == ref_str.lower():
		return True

	## [a, b] vs. [c, d], return a==c and b==d
	if (
		regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
		and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
	):
		pred_parts = prediction[1:-1].split(",")
		ref_parts = reference[1:-1].split(",")
		if len(pred_parts) == len(ref_parts):
			if all(
				[
					math_equal(
						pred_parts[i], ref_parts[i], include_percentage, is_close
					)
					for i in range(len(pred_parts))
				]
			):
				return True
	if (
		(
			prediction.startswith("\\begin{pmatrix}")
			or prediction.startswith("\\begin{bmatrix}")
		)
		and (
			prediction.endswith("\\end{pmatrix}")
			or prediction.endswith("\\end{bmatrix}")
		)
		and (
			reference.startswith("\\begin{pmatrix}")
			or reference.startswith("\\begin{bmatrix}")
		)
		and (
			reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
		)
	):
		pred_lines = [
			line.strip()
			for line in prediction[
				len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
			].split("\\\\")
			if line.strip()
		]
		ref_lines = [
			line.strip()
			for line in reference[
				len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
			].split("\\\\")
			if line.strip()
		]
		matched = True
		if len(pred_lines) == len(ref_lines):
			for pred_line, ref_line in zip(pred_lines, ref_lines):
				pred_parts = pred_line.split("&")
				ref_parts = ref_line.split("&")
				if len(pred_parts) == len(ref_parts):
					if not all(
						[
							math_equal(
								pred_parts[i],
								ref_parts[i],
								include_percentage,
								is_close,
							)
							for i in range(len(pred_parts))
						]
					):
						matched = False
						break
				else:
					matched = False
				if not matched:
					break
		else:
			matched = False
		if matched:
			return True

	if prediction.count("=") == 1 and reference.count("=") == 1:
		pred = prediction.split("=")
		pred = f"{pred[0].strip()} - ({pred[1].strip()})"
		ref = reference.split("=")
		ref = f"{ref[0].strip()} - ({ref[1].strip()})"
		if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
			return True
	elif (
		prediction.count("=") == 1
		and len(prediction.split("=")[0].strip()) <= 2
		and "=" not in reference
	):
		if math_equal(
			prediction.split("=")[1], reference, include_percentage, is_close
		):
			return True
	elif (
		reference.count("=") == 1
		and len(reference.split("=")[0].strip()) <= 2
		and "=" not in prediction
	):
		if math_equal(
			prediction, reference.split("=")[1], include_percentage, is_close
		):
			return True

	# symbolic equal with sympy
	if timeout:
		if call_with_timeout(symbolic_equal_process, prediction, reference):
			return True
	else:
		if symbolic_equal(prediction, reference):
			return True

	## string match for fracs
	prediction = prediction.replace("\\", "")
	reference = reference.replace("\\", "")

	pattern = r"([-+]?\d*\.?\d+)\s*\/\s*([-+]?\d*\.?\d+)"
	pred_match = re.search(pattern, prediction)
	ref_match = re.search(pattern, reference)
	if pred_match and ref_match:
		pred_a, pred_b = pred_match.groups()
		ref_a, ref_b = ref_match.groups()
		if abs(float(pred_a) / float(pred_b) - float(ref_a) / float(ref_b)) < 1e-6:
			prediction = prediction.replace(f"{pred_a}/{pred_b}", f"{ref_a}/{ref_b}")
			return math_equal(
				prediction,
				reference,
				include_percentage,
				is_close,
				recursive_timeout,
				recursive_count + 1,
			)

	return False


def math_equal_process(param):
	return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
	# Note that relative tolerance has significant impact
	# on the result of the synthesized GSM-Hard dataset
	# if reference.is_integer():
	#     return isclose(reference, round(prediction), abs_tol=1e-4)
	# else:
	# prediction = round(prediction, len(str(reference).split(".")[-1]))
	return isclose(reference, prediction, abs_tol=1e-2)


def symbolic_equal(a, b):
	def _parse(s):
		for f in [parse_latex, parse_expr, latex2sympy]:
			try:
				return f(s.replace("\\\\", "\\"))
			except:
				try:
					return f(s)
				except:
					pass
		return s

	a = _parse(a)
	b = _parse(b)

	# direct equal
	try:
		if str(a) == str(b) or a == b:
			return True
	except:
		pass

	# simplify equal
	try:
		if a.equals(b) or simplify(a - b) == 0:
			return True
	except:
		pass

	# equation equal
	try:
		if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
			return True
	except:
		pass

	try:
		if numeric_equal(float(N(a)), float(N(b))):
			return True
	except:
		pass

	# matrix
	try:
		# if a and b are matrix
		if a.shape == b.shape:
			_a = a.applyfunc(lambda x: round(x, 3))
			_b = b.applyfunc(lambda x: round(x, 3))
			if _a.equals(_b):
				return True
	except:
		pass

	return False


def symbolic_equal_process(a, b, output_queue):
	result = symbolic_equal(a, b)
	output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
	output_queue = multiprocessing.Queue()
	process_args = args + (output_queue,)
	process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
	process.start()
	process.join(timeout)

	if process.is_alive():
		process.terminate()
		process.join()
		return False

	return output_queue.get()


def _test_math_equal():
	# print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
	# print(math_equal("(1,4.5)", "(1,\\frac{9}{2})"))
	# print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))
	# print(math_equal("\\sec^2(y)", "\\tan^2(y)+1", timeout=True))
	# print(math_equal("\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\end{pmatrix}", "(\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix})", timeout=True))

	# pred = '\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}'
	# gt = '(\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix})'

	# pred= '-\\frac{8x^2}{9(x^2-2)^{5/3}}+\\frac{2}{3(x^2-2)^{2/3}}'
	# gt= '-\\frac{2(x^2+6)}{9(x^2-2)\\sqrt[3]{x^2-2}^2}'

	# pred =  '-34x-45y+20z-100=0'
	# gt = '34x+45y-20z+100=0'

	# pred = '\\frac{100}{3}'
	# gt = '33.3'

	# pred = '\\begin{pmatrix}0.290243531202435\\\\0.196008371385084\\\\-0.186381278538813\\end{pmatrix}'
	# gt = '(\\begin{pmatrix}0.29\\\\0.196\\\\-0.186\\\\\\end{pmatrix})'

	# pred = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}'
	# gt = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}'

	# pred = '(+5)(b+2)'
	# gt = '(a+5)(b+2)'

	# pred = '\\frac{1+\\sqrt{5}}{2}'
	# gt = '2'

	# pred = '\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}', gt = '4'
	# pred = '1', gt = '1\\\\sqrt{19}'

	# pred = "(0.6,2.6667]"
	# gt = "(\\frac{3}{5},\\frac{8}{3}]"

	# gt = "x+2n+1"
	# pred = "x+1"

	# gt = "0.006"
	# pred = "6 \\times 10^{-3} \\, m/s^2"

	# gt = "np.arcsin(10/13)"
	# pred = "\\arcsin{\\frac{1}{1.3}}"

	# gt = "4e16"
	# pred = "} 4 \\times 10^{16}  erg/s \\]"

	gt = "0.01"
	pred = "0.0122"

	print(math_equal(pred, gt, timeout=True))


if __name__ == "__main__":
	_test_math_equal()
