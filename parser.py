import re
import string
from random import sample
from typing import Any, Dict

import regex
from utils import *
from word2number import w2n


def _fix_fracs(string):
	substrs = string.split("\\frac")
	new_str = substrs[0]
	if len(substrs) > 1:
		substrs = substrs[1:]
		for substr in substrs:
			new_str += "\\frac"
			if len(substr) > 0 and substr[0] == "{":
				new_str += substr
			else:
				try:
					assert len(substr) >= 2
				except:
					return string
				a = substr[0]
				b = substr[1]
				if b != "{":
					if len(substr) > 2:
						post_substr = substr[2:]
						new_str += "{" + a + "}{" + b + "}" + post_substr
					else:
						new_str += "{" + a + "}{" + b + "}"
				else:
					if len(substr) > 2:
						post_substr = substr[2:]
						new_str += "{" + a + "}" + b + post_substr
					else:
						new_str += "{" + a + "}" + b
	string = new_str
	return string


def _fix_a_slash_b(string):
	if not re.match(r"^\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\s*$", string):
		return string
	if len(string.split("/")) != 2:
		return string
	a = string.split("/")[0]
	b = string.split("/")[1]
	# if "sqrt" not in a:
	# 	a = int(a)
	# if "sqrt" not in b:
	# 	b = int(b)
	if string == "{}/{}".format(a, b) and " " not in a:
		new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
		return new_string
	elif " " in a:
		if "and" in a:
			a.replace("and", "")
		
		if not (" " in a and not any(op in a for op in "+-*/=\\")):
			a = strip_string(a)
			b = strip_string(b)
			new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
			return new_string
		k = a.split(" ")[0]
		if len(a.split(" ")) > 1:
			a = a.split(" ")[1].strip()
		if string == "{} {}/{}".format(k, a, b) and a != "":
			k = strip_string(k)
			a = strip_string(a)
			b = strip_string(b)
			new_string = str(k) + "\\frac{" + str(a) + "}{" + str(b) + "}"
			return new_string
		else:
			k = strip_string(k)
			a = strip_string(a)
			b = strip_string(b)
			new_string = "\\frac{" + str(k) + "}{" + str(b) + "}"
			return new_string
	return string


def _fix_sqrt(string):
	# Convert Unicode square root symbol √ to LaTeX format
	_string = re.sub(r"(\d+(?:\.\d+)?)√([^{}\s]+)", r"\1\\sqrt(\2)", string)
	# Fix existing \sqrt commands to have braces
	_string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", _string)
	return _string


def convert_word_number(text: str) -> str:
	try:
		text = str(w2n.word_to_num(text))
	except:
		pass
	return text


# units mainly from MathQA
unit_texts = [
	"east",
	"degree",
	"mph",
	"kmph",
	"ft",
	"m sqaure",
	" m east",
	"sq m",
	"deg",
	"mile",
	"q .",
	"monkey",
	"prime",
	"ratio",
	"profit of rs",
	"rd",
	"o",
	"gm",
	"p . m",
	"lb",
	"tile",
	"per",
	"dm",
	"lt",
	"gain",
	"ab",
	"way",
	"west",
	"a .",
	"b .",
	"c .",
	"d .",
	"e .",
	"f .",
	"g .",
	"h .",
	"t",
	"a",
	"h",
	"no change",
	"men",
	"soldier",
	"pie",
	"bc",
	"excess",
	"st",
	"inches",
	"noon",
	"percent",
	"by",
	"gal",
	"kmh",
	"c",
	"acre",
	"rise",
	"a . m",
	"th",
	"π r 2",
	"sq",
	"mark",
	"l",
	"toy",
	"coin",
	"sq . m",
	"gallon",
	"° f",
	"profit",
	"minw",
	"yr",
	"women",
	"feet",
	"am",
	"pm",
	"hr",
	"cu cm",
	"square",
	"v â € ™",
	"are",
	"rupee",
	"rounds",
	"cubic",
	"cc",
	"mtr",
	"s",
	"ohm",
	"number",
	"kmph",
	"day",
	"hour",
	"minute",
	"min",
	"second",
	"man",
	"woman",
	"sec",
	"cube",
	"mt",
	"sq inch",
	"mp",
	"∏ cm ³",
	"hectare",
	"more",
	"sec",
	"unit",
	"cu . m",
	"cm 2",
	"rs .",
	"rs",
	"kg",
	"g",
	"month",
	"km",
	"m",
	"cm",
	"mm",
	"apple",
	"liter",
	"loss",
	"yard",
	"pure",
	"year",
	"increase",
	"decrease",
	"d",
	"less",
	"Surface",
	"litre",
	"pi sq m",
	"s .",
	"metre",
	"meter",
	"inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=True):
	string = str(string).strip()
	# linebreaks
	string = string.replace("\n", "")

	# right "."
	string = string.rstrip(".")

	# remove inverse spaces
	# replace \\ with \
	string = string.replace("\\!", "")
	# string = string.replace("\\ ", "")
	# string = string.replace("\\\\", "\\")

	# matrix
	string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
	string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
	string = string.replace("bmatrix", "pmatrix")

	# replace tfrac and dfrac with frac
	string = string.replace("tfrac", "frac")
	string = string.replace("dfrac", "frac")
	string = (
		string.replace("\\neq", "\\ne")
		.replace("\\leq", "\\le")
		.replace("\\geq", "\\ge")
	)

	# remove \left and \right
	string = string.replace("\\left", "")
	string = string.replace("\\right", "")
	string = string.replace("\\{", "{")
	string = string.replace("\\}", "}")

	# Remove unit: miles, dollars if after is not none
	_string = re.sub(r"\\text{.*?}$", "", string).strip()
	if _string != "" and _string != string:
		# print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
		string = _string

	if not skip_unit:
		# Remove unit: texts
		for _ in range(2):
			for unit_text in unit_texts:
				# use regex, the prefix should be either the start of the string or a non-alphanumeric character
				# the suffix should be either the end of the string or a non-alphanumeric character
				_string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
				if _string != "":
					string = _string

	# Remove circ (degrees)
	string = string.replace("^{\\circ}", "")
	string = string.replace("^\\circ", "")

	# remove dollar signs
	string = string.replace("\\$", "")
	string = string.replace("$", "")
	string = string.replace("\\(", "").replace("\\)", "")

	# convert word number to digit
	string = convert_word_number(string)

	# replace "\\text{...}" to "..."
	string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
	for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
		string = string.replace(key, "")
	string = string.replace("\\emptyset", r"{}")
	string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

	# remove percentage
	string = string.replace("\\%", "")
	string = string.replace("\%", "")
	string = string.replace("%", "")

	# " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
	string = string.replace(" .", " 0.")
	string = string.replace("{.", "{0.")

	# cdot
	# string = string.replace("\\cdot", "")
	if (
		string.startswith("{")
		and string.endswith("}")
		and string.isalnum()
		or string.startswith("(")
		and string.endswith(")")
		and string.isalnum()
		or string.startswith("[")
		and string.endswith("]")
		and string.isalnum()
	):
		string = string[1:-1]

	# inf
	string = string.replace("infinity", "\\infty")
	if "\\infty" not in string:
		string = string.replace("inf", "\\infty")
	string = string.replace("+\\inity", "\\infty")

	# and
	string = string.replace("and", "")
	string = string.replace("\\mathbf", "")

	# use regex to remove \mbox{...}
	string = re.sub(r"\\mbox{.*?}", "", string)

	# quote
	string.replace("'", "")
	string.replace('"', "")

	# i, j
	if "j" in string and "i" not in string:
		string = string.replace("j", "i")

	# replace a.000b where b is not number or b is end, with ab, use regex
	string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
	string = re.sub(r"(\d+)\.0*$", r"\1", string)

	# if empty, return empty string
	if len(string) == 0:
		return string
	if string[0] == ".":
		string = "0" + string

	# to consider: get rid of e.g. "k = " or "q = " at beginning
	if len(string.split("=")) == 2:
		if len(string.split("=")[0]) <= 2:
			string = string.split("=")[1]

	string = _fix_sqrt(string)


	# \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
	string = _fix_fracs(string)
	# NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
	string = _fix_a_slash_b(string)
	# if " " in string:
	# 	old_string = string.split(" ")
	# 	string = ""
	# 	for s in old_string:
	# 		if re.search(r"(\d|-|/|frac)", s):
	# 			string += s
	# string = string.replace(" ", "")

	return string


def extract_multi_choice_answer(samples, pred_list, data_name, summarizer, pred_str):
	found = False
	choices = samples["question"].split("Choices:\n")[-1].split("\n")
	pred = ""

	for p, pnum in pred_list:
		# pred is not empty, just extract the answer. Match such that the letter is
		# followed by specific punctuation
		if any([summarizer[i] in p for i in range(len(summarizer))]):
			continue
		p = re.sub(r"[\[,.:\]]", "", p).strip()
		if data_name == "mmlu_pro":
			match = re.findall(r"(?:^|[ \(\n\[])[a-jA-J](?=[ \n\)\]]|$)", p)
		elif data_name == "commonsense_qa":
			match = re.findall(r"(?:^|[ \(\n\[])[a-eA-E](?=[ \n\)\]]|$)", p)
		else:
			match = re.findall(r"(?:^|[ \(\n\[])[a-dA-D](?=[ \n\)\]]|$)", p)

		if match and (
			any([summarizer[i] in pred_str for i in range(len(summarizer))])
			or len(pred_list) == 1
		):
			pred = match[0].strip(" (),.:")
			found = True
		else:
			last_pos = -1
			## Check if actual choice sentence is in the predicted string
			## Check which choice appears last in the answer.
			## exact match
			for c in choices:
				ch = c.split(")")
				try:
					ch, gold_ans = ch[0].strip("("), ch[1].strip()
					gold_ans = re.sub(r"[,.: ]", "", gold_ans).strip().lower()
					# print(gold_ans, p)
					# print(gold_ans in p.lower())
					p = re.sub(r"\'s", "ary", p).lower()
					p = re.sub(r"[ ]", "", p).strip().lower()
					if gold_ans in p.lower() or p.lower() in gold_ans:
						position = p.rfind(gold_ans)
						if position > last_pos:
							pred = ch
							last_pos = position
							found = True
				except Exception as e:
					continue

			# ## flexible match
			# if not found:
			# 	for c in choices:
			# 		ch = c.split(")")
			# 		ch, gold_ans = ch[0].strip("("), ch[1].strip()
			# 		gold_ans = re.sub(r"[,.: ]", "", gold_ans).strip().lower()
			# 		# print(gold_ans, p)
			# 		# print(gold_ans in p.lower())
			# 		p = re.sub(r"\'s", "ary", p).lower()
			# 		p = re.sub(r"[ ]", "", p).strip().lower()
			# 		if gold_ans in p.lower():
			# 			position = p.rfind(gold_ans)
			# 			if position > last_pos:
			# 				pred = ch
			# 				last_pos = position
			# 				found = True

		if found:
			break

	if not found:
		pred = ""

	return pred


def extract_strategyqa(pred_str):
	pred_str = pred_str.lower()
	last_true_idx = pred_str.rfind("true")
	last_false_idx = pred_str.rfind("false")
	if last_true_idx == -1 and last_false_idx == -1:
		return ""
	if last_true_idx > last_false_idx:
		pred = "true"
	else:
		pred = "false"

	return pred


def extract_math(samples, pred_list, data_name, summarizer, pred_str):
	# extract the last number in the string
	# print(pred_list)
	pred = ""
	for p, _ in pred_list:
		## Search for equations and fractions first
		p = re.sub(r"(\[|\]|<?/?atok>?|the|final|answer|therefore|is|so|thus|\n)", "", p, flags=re.IGNORECASE)
		if "=" in p:
			p = p.split("=")[-1].strip()
		p = re.sub(r"\$\$|\$\\\$", "$", p)
		p = find_dollar(p)
		p = find_box(p)

		p = strip_string(p)
		p = clean_units(p, data_name)
		# print("P=", p)
		# if " " in p_extract:
		# 	p_extract = p.split(" ")[0]

		## remove alphabets and words followed by non-word non-space
		# print("EXtracted = ", p_extract)

		# if we don't extract anything, use the entire string
		## make sure extracted pattern has a single number of any form. If not, assume it is an equation

		if not any(op in p for op in "+-*/=\\"):
			p_extract_num = re.split(r"[^\w\s][a-zA-Z]", p)[0]
			pattern = r"(?:-\$|\$-|-|\$)?\d*[.:/]?\d+(?:[eE][+-]?\d+)?(?:\^[+-]?\d+)?"
			p_extract_num_no_spaces = p_extract_num.replace(" ", "")
			matches = re.findall(pattern, p_extract_num_no_spaces)

			if len(matches) == 1 and matches[0] == p_extract_num_no_spaces:
				p = p_extract_num_no_spaces

		p = p.strip("[] ;:.,")
		if p != "" and not all(c in string.punctuation or c.isspace() for c in p):
			pred = p
			break

	## use last number
	if pred == "":
		if pred == "":  # use the last number
			pattern = r"(?:-\$|\$-|-|\$)?\d*[.:/]?\d+(?:[eE][+-]?\d+)?(?:\^[+-]?\d+)?"
			pred = re.findall(pattern, pred_str.replace(",", ""))
			if len(pred) >= 1:
				pred = pred[-1]
			else:
				pred = ""

		if "=" in pred:
			pred = pred.split("=")[-1].strip()

		pred = re.sub(r"\$\$|\$\\\$", "$", pred)
		pred = find_dollar(pred)
		pred = find_box(pred)
		# if (
		# 	data_name in ["gsm8k", "math"]
		# 	and "/" not in pred
		# 	and ":" not in pred
		# 	and "frac" not in pred
		# ):
		# 	pred_stripped = re.findall(r"-?\d*\.?\d+", re.sub(r"[, ]", "", pred))
		# 	if len(pred_stripped) >= 1:
		# 		pred = pred_stripped[-1].strip("./ ")

		pred = strip_string(pred)
		pred = clean_units(pred, data_name)
		if " " in pred:
			pred = pred.split(" ")[0]

		pred = pred.strip(".: \n")

	if pred == "":
		try:
			pred = extract_multi_choice_answer(
				samples, pred_list, data_name, summarizer, pred_str
			)
		except Exception as e:
			pred = ""

	return pred


def extract_gsm8k(pred_list, data_name):
	found = False
	# extract the last number in the string
	pred = ""
	for p, _ in pred_list:
		## Search for equations and fractions first
		# print("P = ", p)
		p = re.sub(r"(\[|\]|<?/?atok>?|the|final|answer|therefore|is|so|thus|\n)", "", p, flags=re.IGNORECASE)
		if "=" in p:
			p = p.split("=")[-1].strip()
		p_extract = re.sub(r"\$\$|\$\\\$", "$", p)
		p_extract = find_dollar(p_extract)
		p_extract = find_box(p_extract)

		p_extract = strip_string(p_extract)
		p_extract = clean_units(p_extract, data_name)
		if " " in p_extract:
			p_extract = p.split(" ")[0]
		p_extract = re.split(r"[^\w\s][a-zA-Z]", p_extract)[0]

		## remove alphabets and words followed by non-word non-space
		# print("EXtracted = ", p_extract)

		# if we don't extract anything, try to extract the last number
		if p_extract == "" or not re.search(r"\d", p_extract):
			pattern = r"(?:-\$|\$-|-|\$)?\d*[.:/]?\d+(?:[eE][+-]?\d+)?(?:\^[+-]?\d+)?"
			p = re.findall(pattern, p.replace(",", ""))
			if len(p) >= 1:
				p = p[-1]
			else:
				p = ""
		else:
			p = p_extract

		# print("Cleaned = ", p)
		if p != "":
			pred = p
			found = True
			break

	return pred


direct_answer_trigger_for_fewshot = ("choice is", "answer is", "<atok>")


def choice_answer_clean(pred: str):
	pred = pred.strip("</atok")
	pred = pred.strip("\n")

	# Determine if this is ICL, if so, use \n\n to split the first chunk.
	ICL = False
	for trigger in direct_answer_trigger_for_fewshot:
		if pred.count(trigger) > 1:
			ICL = True
	if ICL:
		pred = pred.split("\n\n")[0]

	# Split the trigger to find the answer.
	preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
	if len(preds) > 1:
		answer_flag = True
		pred = preds[-1]
	else:
		answer_flag = False

	pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

	# Clean the answer based on the dataset
	tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
	if tmp:
		pred = tmp
	else:
		pred = [pred.strip().strip(".")]

	if len(pred) == 0:
		pred = ""
	else:
		if answer_flag:
			# choose the first element in list ...
			pred = pred[0]
		else:
			# choose the last e
			pred = pred[-1]

	# Remove the period at the end, again!
	pred = pred.rstrip(".").rstrip("/")

	return pred


def find_box(pred_str: str):
	if "boxed" not in pred_str:
		return pred_str
	ans = pred_str.split("boxed")[-1]
	if not ans:
		return ""
	if ans[0] == "{":
		stack = 1
		a = ""
		for c in ans[1:]:
			if c == "{":
				stack += 1
				a += c
			elif c == "}":
				stack -= 1
				if stack == 0:
					break
				a += c
			else:
				a += c
	else:
		a = ans.split("$")[0].strip()
	return a


def find_dollar(pred_str: str):
	if pred_str.count("$") < 2:
		return pred_str
	p = 0
	while p < len(pred_str) and pred_str[p] != "$":
		p += 1

	stack = 1
	a = ""
	for c in pred_str[p + 1 :]:
		if c == "$":
			stack -= 1
			if stack == 0:
				break
			a += c
		else:
			a += c
	return a


def clean_units(pred_str: str, data_name: str):
	"""Clean the units in the number."""

	def convert_pi_to_number(code_string):
		code_string = code_string.replace("\\pi", "π")
		# Replace \pi or π not preceded by a digit or } with 3.14
		code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
		# Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
		code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
		# Handle cases where π is within braces or followed by a multiplication symbol
		# This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
		code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
		code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
		return code_string

	if not any(
		x in data_name
		for x in [
			"math",
			"math500",
			"gaokao2023en",
			"college_math",
			"agieval_math",
			"minerva_math",
		]
	):
		pred_str = convert_pi_to_number(pred_str)
	else:
		pred_str = pred_str.replace("π", "\\pi")
	pred_str = pred_str.replace("%", "/100")
	pred_str = pred_str.replace("$", "")
	pred_str = pred_str.replace("¥", "")
	pred_str = pred_str.replace("°C", "")
	pred_str = pred_str.replace(" C", "")
	pred_str = pred_str.replace("°", "")
	return pred_str


# def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
# 	if any([option in pred.lower() for option in ["yes", "true"]]):
# 		pred = "True"
# 	elif any([option in pred.lower() for option in ["no", "false"]]):
# 		pred = "False"
# 	elif any(
# 		[
# 			option in pred.lower()
# 			for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# 		]
# 	):
# 		pass
# 	else:
# 		# Some of the models somehow get used to boxed output from pre-training
# 		if "boxed" in pred:
# 			pred = find_box(pred)

# 		if answer_flag:
# 			# Extract the numbers out of the string
# 			pred = pred.split("=")[-1].strip()
# 			pred = clean_units(pred)
# 			try:
# 				tmp = str(latex2sympy(pred))
# 				pred = str(eval(tmp))
# 			except Exception:
# 				if re.match(r"-?[\d\.]+\s\D+$", pred):
# 					pred = pred.split(" ")[0]
# 				elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
# 					pred = pred.split(" ")[0]
# 		else:
# 			# desparate search over the last number
# 			preds = re.findall(r"-?\d*\.?\d+", pred)
# 			if len(preds) >= 1:
# 				pred = preds[-1]
# 			else:
# 				pred = ""

# 	return pred


#
def extract_answer(pred_str, data_name, samples, use_last_number=True):
	pred_str = pred_str.replace("\u043a\u0438", "")

	# if data_name in ["ifeval"]:
	# 	return pred_str

	# elif "mgsm" in data_name:
	# 	match = re.findall(r"-?[\d.,]+", pred_str)
	# 	if len(match) >= 1:
	# 		return re.sub(r"[^\d\-.]", "", match[-1].strip(".,!"))
	# 	return ""

	# elif data_name == "pop_qa":
	# 	return pred_str

	# elif data_name in ["winogrande", "commonsense_qa"]:
	# 	pred = ""
	# 	pred_str = re.sub(r"[,.:]", "", pred_str).strip()
	# 	match = re.findall(r"(?:^|[ \(\n])[a-dA-D](?=[ \n\)]|$)", pred_str)
	# 	found = False
	# 	choices = samples["question"].split("Choices:\n")[-1].split("\n")
	# 	if match:
	# 		pred = match[0].strip(" (),.:")
	# 		found = True
	# 	else:
	# 		last_pos = -1
	# 		## Check if actual choice sentence is in the predicted string
	# 		## Check which choice appears last in the answer.
	# 		## exact match
	# 		for c in choices:
	# 			ch = c.split(")")
	# 			ch, gold_ans = ch[0].strip("("), ch[1].strip()
	# 			gold_ans = re.sub(r"[,.: ]", "", gold_ans).strip().lower()
	# 			if gold_ans in pred_str.lower():
	# 				position = pred_str.rfind(gold_ans)
	# 				if position > last_pos:
	# 					pred = ch
	# 					last_pos = position
	# 					found = True

	# 		## flexible match
	# 		if not found:
	# 			for c in choices:
	# 				ch = c.split(")")
	# 				ch, gold_ans = ch[0].strip("("), ch[1].strip()
	# 				gold_ans = re.sub(r"[,.: ]", "", gold_ans).strip().lower()
	# 				pred_str = re.sub(r"[ ]", "", pred_str).strip().lower()
	# 				if gold_ans in pred_str.lower():
	# 					position = pred_str.rfind(gold_ans)
	# 					if position > last_pos:
	# 						pred = ch
	# 						last_pos = position
	# 						found = True

	if "I hope" in pred_str:
		pred_str = pred_str.split("I hope")[0].strip()


	if "xquad" in data_name:
		return pred_str

	elif data_name in [
		"minerva_math",
		"gaokao2023en",
		"college_math",
		"agieval_math",
		"olympiadbench",
		"deepmind_math",
	]:

		def split_string(s, tok):
			result = []
			parts = s.split(tok)
			for part in reversed(parts):
				result.append(part)
			return result

		def splitAll(string_list, tok):
			result = []
			for s in string_list:
				result.extend(split_string(s, tok))
			return result

		pred_list = [pred_str]
		ll = len(pred_list)
		while True:
			if any("<atok>" in x for x in pred_list):
				pred_list = splitAll(pred_list, "<atok>")
			if any("final answer is" in x for x in pred_list):
				pred_list = splitAll(pred_list, "final answer is")
			if any("answer is" in x for x in pred_list):
				pred_list = splitAll(pred_list, "answer is")
			if any("answer to the question is" in x for x in pred_list):
				pred_list = splitAll(pred_list, "answer to the question is")
			if any("Final Answer" in x for x in pred_list):
				pred_list = splitAll(pred_list, "Final Answer")
			if any("choice is" in x for x in pred_list):
				pred_list = splitAll(pred_list, "choice is")
			if any("choice:" in x for x in pred_list):
				pred_list = splitAll(pred_list, "choice:")
			if any("####" in x for x in pred_list):
				pred_list = splitAll(pred_list, "####")
			if len(pred_list) == ll:
				break
			ll = len(pred_list)

		pred = ""
		num_ans = False
		char_ans = False
		for c in samples["gt"]:
			if c.isdigit():
				num_ans = True
				break
			if c.isalpha():
				char_ans = True


		for p in pred_list:
			p = find_box(p)
			if ".\n" in p:
				if re.search(r"(atok|final answer is|boxed)", p.split(".\n")[-1]):
					p = p.split(".\n")[-1].strip()
				else:
					p = p.split(".\n")[0].strip()
			if ". " in p:
				if re.search(r"(atok|final answer is|boxed)", p.split(". ")[-1]):
					p = p.split(". ")[-1].strip()
				else:
					p = p.split(". ")[0].strip()
			if "I hope" in p:
				p = p.split("I hope")[0].strip()
			# if "\n" in p:
			# 	p = p.split("\n")[0]
			p = re.sub(r"(the|final|answer|is|<?/?atok>?|\n)", "", p, flags=re.IGNORECASE)
			if "=" in p:
				p = p.split("=")[-1].strip()
			p = re.sub(r"\$\$|\$\\\$", "$", p)			
			p = find_dollar(p)
			p = strip_string(p)
			p = clean_units(p, data_name)			

			has_num = False
			for c in p:
				if c.isdigit():
					has_num = True
					break
			if num_ans and not has_num:
				continue

			if num_ans and not char_ans and not any(x in p for x in ['\\times', '\\div', '\\frac', '^', 'sqrt', 'log']):
				pattern = (
					r"(?:-\$|\$-|-|\$)?\d*[.:/]?\d+(?:[eE][+-]?\d+)?(?:\^[+-]?\d+)?"
				)
				p = re.findall(pattern, re.sub(r"[,:\[\]]", "", pred_str))
				if len(p) >= 1:
					p = p[-1]
				else:
					p = ""
				p = p.strip()



			# if " " in p and any(
			# 	x in p.lower() for x in ["you", "question", "final", "i ", "i'm"]
			# ):
			# 	ps_list = p.split(" ")
			# 	for ps in ps_list:
			# 		has_num = False
			# 		for c in ps:
			# 			if c.isdigit():
			# 				has_num = True
			# 				break
			# 		if not num_ans:
			# 			p = ps
			# 			break
			# 		elif num_ans and has_num:
			# 			p = ps
			# 			break
			p = p.strip(".: \n")
			if p == "":  # use the last number
				pattern = (
					r"(?:-\$|\$-|-|\$)?\d*[.:/]?\d+(?:[eE][+-]?\d+)?(?:\^[+-]?\d+)?"
				)
				p = re.findall(pattern, re.sub(r"[,:\[\]]", "", pred_str))
				if len(p) >= 1:
					p = p[-1]
				else:
					p = ""
				p = p.strip()
			if p != "":
				pred = p
				break

	elif data_name in [
		"arc_c",
		"strategyqa",
		"gpqa",
		"commonsense_qa",
		"gsm8k",
		"svamp",
		"math",
		"math500",
		"mmlu_stem",
		"mmlu_pro",
		"aime24",
		"amc23",
	]:
		summarizer = [
			# split by keywords for new sentence
			"boxed",
			"<atok>",
			"atok",
			"answer",
			"answer is",
			"answer to the question is",
			"Final Answer",
			"choice is",
			"choice:",
			"####",
			"Therefore",
			"Thus",
			"correct answer",
			"best answer",
		]

		# punctuation = [
		# 	# ## split by punctuation
		# 	"\n",
		# 	"So",
		# 	". ",
		# ]

		skip_strings = [
			"final",
			"answer",
			"atok",
			"correct",
			"solution",
			"is",
			"thus",
			"the",
			"final",
			"therefore",
		]

		pred_list = [[pred_str, 2]]
		# search if the string can be split further
		final_pred_list = []
		while len(pred_list) > 0:
			found_string = False
			p, pnum = pred_list.pop(0)
			for summarize_string in summarizer:
				if summarize_string in p:
					found_string = True
					p = p.strip(".,\n !?>/<")
					p = p.split(summarize_string)
					for i in p:
						si = i.strip(".,\n !?>/<")
						pred_list.append([si, 0])
					break

			## Only atomic strings added
			if not found_string:
				sstrip = p.strip(string.punctuation).strip()
				for sskip in skip_strings:
					if (
						sskip.lower() == sstrip.lower()
						or sstrip == ""
						and all(c in string.punctuation or c.isspace() for c in sstrip)
					):
						found_string = True
						break

				if not found_string:
					final_pred_list.append([p, pnum])

		pred_list = final_pred_list
		pred_list = [pl for pl in sorted(pred_list, key=lambda x: (x[1], len(x[0])))]

		# remove words, just take the number
		pred = ""

		if data_name in ["strategyqa"]:
			pred = extract_strategyqa(
				pred_str,
			)
		elif data_name in ["gpqa", "arc_c", "mmlu_stem", "mmlu_pro", "commonsense_qa"]:
			pred = extract_multi_choice_answer(
				samples, pred_list, data_name, summarizer, pred_str
			)
		elif data_name in ["gsm8k", "aime24", "amc23", "svamp"]:
			pred = extract_gsm8k(pred_list, data_name)
		elif any(x in data_name for x in ["math", "math500"]):
			pred = extract_math(samples, pred_list, data_name, summarizer, pred_str)

		else:
			raise ValueError(f"Unknown data name: {data_name}")
	return pred


STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


def parse_ground_truth(example: Dict[str, Any], data_name):
	if "gt_cot" in example and "gt" in example:
		if data_name in STRIP_EXCEPTIONS:
			gt_ans = example["gt"]
		else:
			gt_ans = strip_string(example["gt"])
		return example["gt_cot"], gt_ans

	if data_name == "pop_qa":
		gt_cot = None
		gt_ans = example["possible_answers"]
	elif "minerva_math" in data_name:
		gt_cot = example["solution"]
		gt_ans = find_box(gt_cot)
	elif any(
		x in data_name
		for x in [
			"gaokao2023en",
			"college_math",
			"gaokao_math_cloze",
			"agieval_math",
			"aime24",
			"amc23",
		]
	):
		gt_cot, gt_ans = None, example["answer"]
	elif data_name == "olympiadbench":
		gt_cot, gt_ans = None, str(example["final_answer"][0]).strip("$")
	elif "gpqa" in data_name:
		gt_cot = example["explanation"]
		gt_ans = example["answer"]
	elif "gsm8k" in data_name:
		gt_cot, gt_ans = example["answer"].split("####")
	elif "xquad" in data_name:
		gt_ans = example["answer"][0]
		gt_cot = None
	elif "svamp" in data_name or "deepmind_math" in data_name:
		gt_cot = None
		gt_ans = example["answer"]
	elif any(x in data_name for x in ["math", "math500"]):  # 关键代码
		gt_cot = example["solution"]
		gt_ans = example["answer"]
	elif "arc_c" in data_name:
		gt_ans = example["answer"]
		gt_cot = None
	elif "strategyqa" in data_name:
		gt_ans = example["answer"]
		gt_cot = None
	elif "mgsm" in data_name:
		gt_cot = example["gt_cot"]
		gt_ans = example["answer"]
	elif any(x in data_name for x in ["commonsense_qa"]):
		gt_cot = None
		gt_ans = example["answer"]
	elif "mmlu_stem" in data_name:
		abcd = "ABCD"
		gt_cot, gt_ans = None, abcd[example["answer"]]
	elif "mmlu_pro" in data_name:
		gt_cot, gt_ans = example["cot_content"], example["answer"]
	else:
		raise NotImplementedError(f"`{data_name}`")
	gt_cot = str(gt_cot).strip()
	if "pop_qa" in data_name:
		pass
	elif data_name not in STRIP_EXCEPTIONS:
		gt_ans = strip_string(gt_ans, skip_unit=data_name == "carp_en")
	else:
		gt_ans = (
			gt_ans.replace("\\neq", "\\ne")
			.replace("\\leq", "\\le")
			.replace("\\geq", "\\ge")
		)

	return gt_cot, gt_ans


def parse_question(example, data_name):
	question = ""
	if data_name == "asdiv":
		question = f"{example['body'].strip()} {example['question'].strip()}"
	elif data_name == "svamp":
		body = example["Body"].strip()
		if not body.endswith("."):
			body = body + "."
		question = f"{body} {example['Question'].strip()}"
	elif data_name == "tabmwp":
		title_str = (
			f'regarding "{example["table_title"]}" ' if example["table_title"] else ""
		)
		question = f"Read the following table {title_str}and answer a question:\n"
		question += f"{example['table']}\n{example['question']}"
		if example["choices"]:
			question += (
				f" Please select from the following options: {example['choices']}"
			)
	elif data_name == "carp_en":
		question = example["content"]
	elif data_name == "mmlu_stem":
		options = example["choices"]
		assert len(options) == 4
		for i, (label, option) in enumerate(zip("ABCD", options)):
			options[i] = f"({label}) {str(option).strip()}"
		options = " ".join(options)
		# question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
		question = f"{example['question'].strip()}\nAnswer Choices: {options}"
	elif data_name == "sat_math":
		options = example["options"].strip()
		assert "A" == options[0]
		options = "(" + options
		for ch in "BCD":
			if f" {ch}) " in options:
				options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
		# question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
		question = f"{example['question'].strip()}\nAnswer Choices: {options}"
	elif "aqua" in data_name:
		options = example["options"]
		choice = "(" + "(".join(options)
		choice = choice.replace("(", " (").replace(")", ") ").strip()
		choice = "\nAnswer Choices: " + choice
		question = example["question"].strip() + choice
	elif data_name == "gaokao_math_qa":
		options_dict = example["options"]
		options = []
		for key in options_dict:
			options.append(f"({key}) {options_dict[key]}")
		options = " ".join(options)
		question = f"{example['question'].strip()}\n选项: {options}"
	else:
		for key in ["question", "problem", "Question", "input"]:
			if key in example:
				question = example[key]
				break
	# assert question != ""
	# Yes or No question
	_, gt_ans = parse_ground_truth(example, data_name)
	if isinstance(gt_ans, str):
		gt_lower = gt_ans.lower()
		if gt_lower in ["true", "false"]:
			question += " (True or False)"
		if gt_lower in ["yes", "no"]:
			question += " (Yes or No)"
	return question.strip()


def run_execute(executor, result, prompt_type, data_name, execute=False, samples=None):
	if not result or result == "error":
		return None, None
	report = None

	# if "program_only" in prompt_type:
	# 	prediction = extract_program_output(result)
	# elif prompt_type in ["pot", "pal"] and execute:
	# 	code = extract_program(result)
	# 	prediction, report = executor.apply(code)
	# elif prompt_type == "gpqa":
	# 	prediction = extract_multi_choice_answer(result)
	# if "ifeval" in data_name:
	# 	prediction = result
	# else:
	prediction = extract_answer(result, data_name, samples)

	# prediction = strip_string(prediction, skip_unit=data_name == "carp_en")
	# prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)
	return prediction, report


def _test_extract_answer():
	sample = {"idx": 268, "question": "Preamble: Consider the Particle in an Infinite Box ``superposition state'' wavefunction,\n\\[\n\\psi_{1,2}=(1 / 3)^{1 / 2} \\psi_{1}+(2 / 3)^{1 / 2} \\psi_{2}\n\\]\nwhere $E_{1}$ is the eigen-energy of $\\psi_{1}$ and $E_{2}$ is the eigen-energy of $\\psi_{2}$.\n\nSubproblem 0: Suppose you do one experiment to measure the energy of $\\psi_{1,2}$.  List the possible result(s) of your measurement.\n\n\nSolution: Since the only eigenergies are $E_{1}$ and $E_{2}$, the possible outcomes of the measurement are $\\boxed{E_{1},E_{2}}$.\n\nFinal answer: The final answer is E_{1},E_{2}. I hope it is correct.\n\nSubproblem 1: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What average energy will you observe?", "gt_cot": "\\[\n\\langle E\\rangle =\\boxed{\\frac{1}{3} E_{1}+\\frac{2}{3} E_{2}}\n\\]\nThis value of $\\langle E\\rangle$ is between $E_{1}$ and $E_{2}$ and is the weighted average energy.", "gt": "\\frac{1}{3} E_{1}+\\frac{2}{3} E_{2}", "prompt": "Reason step by step and give a final answer to the following question. Your response should always end with \"The final answer is <atok> [answer] </atok>.\" where [answer] is the correct solution to the problem.\n\nQuestion:\nPreamble: Consider the Particle in an Infinite Box ``superposition state'' wavefunction,\n\\[\n\\psi_{1,2}=(1 / 3)^{1 / 2} \\psi_{1}+(2 / 3)^{1 / 2} \\psi_{2}\n\\]\nwhere $E_{1}$ is the eigen-energy of $\\psi_{1}$ and $E_{2}$ is the eigen-energy of $\\psi_{2}$.\n\nSubproblem 0: Suppose you do one experiment to measure the energy of $\\psi_{1,2}$.  List the possible result(s) of your measurement.\n\n\nSolution: Since the only eigenergies are $E_{1}$ and $E_{2}$, the possible outcomes of the measurement are $\\boxed{E_{1},E_{2}}$.\n\nFinal answer: The final answer is E_{1},E_{2}. I hope it is correct.\n\nSubproblem 1: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What average energy will you observe?\n\nAnswer: Let's think step by step.\n", "type": "Physical Chemistry (5.61 Fall 2017)", "solution": "\\[\n\\langle E\\rangle =\\boxed{\\frac{1}{3} E_{1}+\\frac{2}{3} E_{2}}\n\\]\nThis value of $\\langle E\\rangle$ is between $E_{1}$ and $E_{2}$ and is the weighted average energy.", "code": ["The average energy is given by the expectation value of the energy operator, which is the Hamiltonian $H$. The expectation value of $H$ for a state $\\psi$ is given by $\\langle \\psi | H | \\psi \\rangle$.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. The Hamiltonian $H$ is diagonal in the basis of eigenstates $\\psi_1$ and $\\psi_2$, so $H \\psi_1 = E_1 \\psi_1$ and $H \\psi_2 = E_2 \\psi_2$.\n\nThe expectation value of $H$ for $\\psi_{1,2}$ is:\n\\[\n\\langle \\psi_{1,2} | H | \\psi_{1,2} \\rangle = \\left( \\frac{1}{3} \\right)^{1/2} \\langle \\psi_1 | H | \\psi_{1,2} \\rangle + \\left( \\frac{2}{3} \\right)^{1/2} \\langle \\psi_2 | H | \\psi_{1,2} \\rangle\n\\]\n\\[\n= \\left( \\frac{1}{3} \\right)^{1/2} \\langle \\psi_1 | E_1 \\psi_1 \\rangle + \\left( \\frac{2}{3} \\right)^{1/2} \\langle \\psi_2 | E_2 \\psi_2 \\rangle\n\\]\n\\[\n= \\left( \\frac{1}{3} \\right)^{1/2} E_1 \\langle \\psi_1 | \\psi_1 \\rangle + \\left( \\frac{2}{3} \\right)^{1/2} E_2 \\langle \\psi_2 | \\psi_2 \\rangle\n\\]\n\\[\n= \\left( \\frac{1}{3} \\right)^{1/2} E_1 + \\left( \\frac{2}{3} \\right)^{1/2} E_2\n\\]\n\\[\n= \\frac{1}{3} E_1 + \\frac{2}{3} E_2\n\\]\n\nSo, the average energy observed is $\\boxed{\\frac{1}{3} E_1 + \\frac{2}{3} E_2}$.\n\nFinal answer: The final answer is \\frac{1}{3} E_{1} + \\frac{2}{3} E_{2}. I hope it is correct.\n\nSubproblem 2: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{1}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. The coefficient of $\\psi_1$ is $(1/3)^{1/2}$, so the probability of observing $E_1$ is:\n\\[\n\\left| \\frac{1}{3} \\right|^{1/2} = \\frac{1}{3}\n\\]\n\nSo, the probability of observing $E_1$ is $\\boxed{\\frac{1}{3}}$.\n\nFinal answer: The final answer is \\frac{1}{3}. I hope it is correct.\n\nSubproblem 3: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{2}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. The coefficient of $\\psi_2$ is $(2/3)^{1/2}$, so the probability of observing $E_2$ is:\n\\[\n\\left| \\frac{2}{3} \\right|^{1/2} = \\frac{2}{3}\n\\]\n\nSo, the probability of observing $E_2$ is $\\boxed{\\frac{2}{3}}$.\n\nFinal answer: The final answer is \\frac{2}{3}. I hope it is correct.\n\nSubproblem 4: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{3}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_3$ in the superposition, so the probability of observing $E_3$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_3$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 5: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{4}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_4$ in the superposition, so the probability of observing $E_4$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_4$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 6: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{5}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_5$ in the superposition, so the probability of observing $E_5$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_5$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 7: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{6}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_6$ in the superposition, so the probability of observing $E_6$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_6$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 8: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{7}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_7$ in the superposition, so the probability of observing $E_7$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_7$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 9: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{8}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_8$ in the superposition, so the probability of observing $E_8$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_8$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 10: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{9}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_9$ in the superposition, so the probability of observing $E_9$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_9$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 11: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{10}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{10}$ in the superposition, so the probability of observing $E_{10}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{10}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 12: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{11}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{11}$ in the superposition, so the probability of observing $E_{11}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{11}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 13: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{12}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{12}$ in the superposition, so the probability of observing $E_{12}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{12}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 14: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{13}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{13}$ in the superposition, so the probability of observing $E_{13}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{13}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 15: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{14}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{14}$ in the superposition, so the probability of observing $E_{14}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{14}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 16: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{15}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{15}$ in the superposition, so the probability of observing $E_{15}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{15}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 17: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{16}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{16}$ in the superposition, so the probability of observing $E_{16}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{16}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 18: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{17}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{17}$ in the superposition, so the probability of observing $E_{17}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{17}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 19: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{18}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2} \\psi_2$. There is no $\\psi_{18}$ in the superposition, so the probability of observing $E_{18}$ is:\n\\[\n0\n\\]\n\nSo, the probability of observing $E_{18}$ is $\\boxed{0}$.\n\nFinal answer: The final answer is 0. I hope it is correct.\n\nSubproblem 20: Suppose you do many identical measurements to measure the energies of identical systems in state $\\psi_{1,2}$. What is the probability of observing $E_{19}$?\n\nAnswer: Let's think step by step.\nThe probability of observing a particular energy is given by the square of the absolute value of the coefficient of the corresponding eigenstate in the superposition.\n\nIn this case, the state is $\\psi_{1,2} = (1/3)^{1/2} \\psi_1 + (2/3)^{1/2}"]}
	print("Extracted")
	print(extract_answer(sample["code"][0], "agieval_math", sample))


def test_math(pred, data_name):
	if "=" in pred:
		pred = pred.split("=")[-1].strip()
	pred = re.sub(r"\$\$|\$\\\$", "$", pred)
	pred = find_dollar(pred)
	pred = find_box(pred)

	pred = strip_string(pred)
	# pred = clean_units(pred, data_name)
	# if " " in pred:
	# 	pred = pred.split(" ")[0]

	print("Final: ", pred)


if __name__ == "__main__":
	_test_extract_answer()
