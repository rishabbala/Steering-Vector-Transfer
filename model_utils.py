"""
https://github.com/allenai/open-instruct
"""

import gc
import random
import re
from datetime import datetime

import torch
from data_loader import load_data
from parser import parse_ground_truth
from tqdm import tqdm
from transformers import (
	AutoTokenizer,
	DynamicCache,
	StoppingCriteria,
	StoppingCriteriaList,
)
from utils import construct_prompt
from safetensors.torch import safe_open
import os



class KeywordsStoppingCriteria(StoppingCriteria):
	def __init__(self, keywords_str, tokenizer):
		StoppingCriteria.__init__(self)
		self.current_context = []
		self.tokenizer = tokenizer
		self.keywords_str = keywords_str

	def __call__(
		self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
	) -> bool:
		if len(self.current_context) == 0:
			self.current_context = [[] for _ in range(input_ids.shape[0])]

		# self.current_context.append(input_ids[0][-1].item())
		sequences_should_be_stopped = []
		for i in range(input_ids.shape[0]):
			_id = input_ids[i][-1].item()
			self.current_context[i].append(_id)
			current_context = self.tokenizer.decode(self.current_context[i])
			should_be_stopped = False
			for word in self.keywords_str:
				if word in current_context:
					should_be_stopped = True
					break
			sequences_should_be_stopped.append(should_be_stopped)
		return all(sequences_should_be_stopped)


class KeyWordsCriteriaTrunc(StoppingCriteria):
	def __init__(self, stop_id_sequences, prompt_length):
		assert isinstance(stop_id_sequences[0], list), (
			"stop_id_sequences should be a list of list of ids"
		)
		self.stop_sequences = stop_id_sequences
		self.prompt_length = prompt_length

	def __call__(
		self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
	) -> bool:
		sequences_should_be_stopped = []
		for i in range(input_ids.shape[0]):
			ids = input_ids[i][self.prompt_length :].tolist()
			should_be_stopped = False
			for stop_sequence in self.stop_sequences:
				if input_ids.shape[0] == 1:
					_ids = ids[-len(stop_sequence) :]
				else:
					_ids = ids
				for j in range(len(_ids), 0, -len(stop_sequence)):
					if _ids[max(j - len(stop_sequence), 0) : j] == stop_sequence:
						should_be_stopped = True
						break
				if should_be_stopped:
					break
			sequences_should_be_stopped.append(should_be_stopped)
		return all(sequences_should_be_stopped)


class KeyWordsCriteria(StoppingCriteria):
	def __init__(self, stop_id_sequences):
		assert isinstance(stop_id_sequences[0], list), (
			"stop_id_sequences should be a list of list of ids"
		)
		self.stop_sequences = stop_id_sequences

	def __call__(
		self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
	) -> bool:
		sequences_should_be_stopped = []
		for i in range(input_ids.shape[0]):
			sequence_should_be_stopped = False
			for stop_sequence in self.stop_sequences:
				if input_ids[i][-len(stop_sequence) :].tolist() == stop_sequence:
					sequence_should_be_stopped = True
					break
			sequences_should_be_stopped.append(sequence_should_be_stopped)
		return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(
	model,
	tokenizer,
	prompts,
	batch_size=1,
	stop_id_sequences=None,
	add_special_tokens=True,
	disable_tqdm=False,
	**generation_kwargs,
):
	generations = []
	if not disable_tqdm:
		progress = tqdm(total=len(prompts), desc="Generating Completions")

	print(generation_kwargs)

	num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
	for i in range(0, len(prompts), batch_size):
		batch_prompts = prompts[i : i + batch_size]
		tokenized_prompts = tokenizer(
			batch_prompts,
			padding="longest",
			return_tensors="pt",
			add_special_tokens=add_special_tokens,
		)
		batch_input_ids = tokenized_prompts.input_ids
		attention_mask = tokenized_prompts.attention_mask

		if model.device.type == "cuda":
			batch_input_ids = batch_input_ids.cuda()
			attention_mask = attention_mask.cuda()

		# try:
		if stop_id_sequences is not None:
			stop_criteria = KeywordsStoppingCriteria(stop_id_sequences, tokenizer)
		batch_outputs = model.generate(
			input_ids=batch_input_ids,
			attention_mask=attention_mask,
			stopping_criteria=None
			if stop_id_sequences is None
			else StoppingCriteriaList([stop_criteria]),
			**generation_kwargs,
		)

		# the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
		# so some outputs still have the stop sequence, which we need to remove.
		# if stop_id_sequences:
		#     for output_idx in range(batch_outputs.shape[0]):
		#         for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
		#             if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
		#                 batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
		#                 break

		# remove the prompt from the output
		# we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
		# we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
		# space is important for some tasks (e.g., code completion).
		batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

		batch_prompts = tokenizer.batch_decode(
			batch_input_ids, skip_special_tokens=True
		)
		# duplicate the prompts to match the number of return sequences
		batch_prompts = [
			prompt for prompt in batch_prompts for _ in range(num_return_sequences)
		]
		batch_generations = [
			output[len(prompt) :]
			for prompt, output in zip(batch_prompts, batch_outputs)
		]

		if stop_id_sequences is not None:
			for idx, prediction in enumerate(batch_generations):
				pred = prediction
				for stop_sequence in stop_id_sequences:
					pred = pred.split(stop_sequence)[0]
					pred = re.sub(r"[^0-9A-Za-z{}]+$", "", pred)
				batch_generations[idx] = pred

		generations += batch_generations

		if not disable_tqdm:
			progress.update(len(batch_prompts) // num_return_sequences)

		del (
			batch_input_ids,
			attention_mask,
			batch_outputs,
			batch_prompts,
			batch_generations,
		)
		gc.collect()

	assert len(generations) == len(prompts) * num_return_sequences, (
		"number of generations should be equal to number of prompts * num_return_sequences"
	)
	return generations


@torch.no_grad()
def load_hf_lm_and_tokenizer(
	model_name_or_path,
	device_map="auto",
	use_fast_tokenizer=False,
	padding_side="left",
	dtype=torch.bfloat16,
	steering_vector=None,
	alpha=0.0,
	transfer=False,
	use_flash_attn=True,
):
	if (
		model_name_or_path.lower() == "qwen/qwen3-14b-thinking"
		or model_name_or_path.lower() == "qwen/qwen3-8b-thinking"
	):
		model_name_or_path = "Qwen/Qwen3-8B-Base"
		tokenizer = AutoTokenizer.from_pretrained(
			model_name_or_path,
			use_fast=use_fast_tokenizer,
			padding_side=padding_side,
			trust_remote_code=True,
			enable_thinking=True,
		)
	else:
		tokenizer = AutoTokenizer.from_pretrained(
			model_name_or_path,
			use_fast=use_fast_tokenizer,
			padding_side=padding_side,
			trust_remote_code=True,
		)

	# set pad token to eos token if pad token is not set
	if tokenizer.pad_token is None:
		if tokenizer.unk_token:
			tokenizer.pad_token = tokenizer.unk_token
			tokenizer.pad_token_id = tokenizer.unk_token_id
		elif tokenizer.eos_token:
			tokenizer.pad_token = tokenizer.eos_token
			tokenizer.pad_token_id = tokenizer.eos_token_id
		else:
			raise ValueError(
				"You are using a new tokenizer without a pad token."
				"This is not supported by this script."
			)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.unk_token
		tokenizer.pad_token_id = tokenizer.unk_token_id

	# if "gemma" in model_name_or_path.lower():
	# 	use_flash_attn = False

	# from transformers import AutoModelForCausalLM

	# if not transfer:
	# 	model = AutoModelForCausalLM.from_pretrained(
	# 		model_name_or_path,
	# 		torch_dtype=dtype,
	# 		device_map=device_map,
	# 		trust_remote_code=True,
	# 		attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
	# 	)

	# else:
	if any(
		x in model_name_or_path.lower()
		for x in [
			"qwen2",
			"qwen-2",
			"eurus",
			"qwen1.5",
			"dler-r1",
			"openreasoning-nemotron",
			"sailor",
		]
	):
		from models.qwen2 import Qwen2ForCausalLM

		model = Qwen2ForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif any(
		x in model_name_or_path.lower()
		for x in ["nemotron-cascade-8b", "qwen3", "deepseek-r1-0528-qwen3", "qwen-3"]
	):
		from models.qwen3 import Qwen3ForCausalLM

		model = Qwen3ForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif any(x in model_name_or_path.lower() for x in ["llama", "vicuna", "orca"]):
		from models.llama import LlamaForCausalLM

		print(f"Loading {model_name_or_path}")
		model = LlamaForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)
		print(model)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif "phi-3" in model_name_or_path.lower():
		from models.phi3 import Phi3ForCausalLM

		print(f"Loading {model_name_or_path}")
		model = Phi3ForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif "gemma-2" in model_name_or_path.lower() or "freedomintelligence" in model_name_or_path.lower():
		from models.gemma2 import Gemma2ForCausalLM

		print(f"Loading {model_name_or_path}")
		model = Gemma2ForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif (
		"gemma-3" in model_name_or_path.lower() or "rnj-1" in model_name_or_path.lower()
	):
		from models.gemma3 import Gemma3ForCausalLM

		print(f"Loading {model_name_or_path}")
		model = Gemma3ForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif "olmo-2" in model_name_or_path.lower():
		from models.olmo2 import Olmo2ForCausalLM

		print(f"Loading {model_name_or_path}")
		model = Olmo2ForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif "olmo-3" in model_name_or_path.lower():
		from models.olmo3 import Olmo3ForCausalLM

		print(f"Loading {model_name_or_path}")
		model = Olmo3ForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif "granite-3" in model_name_or_path.lower():
		from models.granite import GraniteForCausalLM

		print(f"Loading {model_name_or_path}")
		model = GraniteForCausalLM.from_pretrained(
			model_name_or_path,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	elif "ministral-3" in model_name_or_path.lower():
		
		model_weights = {}

		pth = os.getenv("HF_DATASETS_CACHE")
		print("pth=", os.path.join(pth, f"models--{model_name_or_path.replace('/', '--')}/snapshots"))
		for r, _, fff in os.walk(os.path.join(pth, f"models--{model_name_or_path.replace('/', '--')}/snapshots")):
			ff = [os.path.join(r, x) for x in fff]
			for f in ff:
				print("f=", f)
				if f.endswith(".safetensors"):
					with safe_open(f, framework="pt", device="cpu") as file:
						for key in file.keys():
							print("All keys =", key)
							if "language_model" in key:
								model_weights[key.replace("language_model.", "")] = file.get_tensor(key)


		for w in model_weights.keys():
			print("Initialized weights =", w)

		from transformers import AutoConfig
		from models.ministral3 import Ministral3ForCausalLM
		from accelerate import load_checkpoint_and_dispatch
		from accelerate.utils import get_balanced_memory

		config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True).text_config
		config.dtype = dtype

		model = Ministral3ForCausalLM.from_pretrained(
			pretrained_model_name_or_path=None,
			config=config,
			state_dict=model_weights,
			torch_dtype=dtype,
			device_map=device_map,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
		)

		for k, v in model.named_parameters():
			print(k, v.device)
		
		# ## Mistral 3 uses Mistral3Model as the language model and ties the lm_head with the embedding_weights.
		model.model.steering_vector = steering_vector
		model.model.transfer = transfer
		model.model.alpha = alpha

	else:
		raise NotImplementedError

	model.eval()
	return model, tokenizer


def prepare_data(data_name, prompt_type, args, reverse=False, num_shots=0):
	examples = load_data(data_name, getattr(args, "split", None), args.data_dir)

	if args.num_samples > 0:
		if reverse:
			examples = examples[-(min(args.num_samples, len(examples))) :]
		else:
			examples = examples[: min(args.num_samples, len(examples))]

	# shuffle
	if args.shuffle:
		random.seed(datetime.now().timestamp())
		random.shuffle(examples)

	samples = []
	for example in tqdm(examples, total=len(examples)):
		if "ifeval" in data_name:
			sample = example
			sample["question"] = sample["prompt"]
			sample["prompt"] = construct_prompt(example, data_name, prompt_type, args)
			samples.append(example)

		else:
			idx = example["idx"]
			gt_cot, gt_ans = parse_ground_truth(example, data_name)
			# example["gt_ans"] = gt_ans
			full_prompt = construct_prompt(example, data_name, prompt_type, args, num_shots=num_shots)

			if idx == 0:
				print(full_prompt)

			sample = {
				"idx": idx,
				"question": example["question"],
				"gt_cot": gt_cot,
				"gt": gt_ans,
				"prompt": full_prompt,
			}

			# add remain fields
			for key in [
				"level",
				"type",
				"unit",
				"solution_type",
				"choices",
				"solution",
				"ques_type",
				"ans_type",
				"answer_type",
				"dataset",
				"subfield",
				"filed",
				"theorem",
				"answer",
			]:
				if key in example:
					sample[key] = example[key]
			samples.append(sample)

		# else:
		# 	raise ValueError(f"Unsupported data name: {data_name}")

	return samples


# if __name__ == "__main__":
# 	_test_generate_completions()

# def load_hf_tokenizer(
# 	model_name_or_path,
# 	tokenizer_name_or_path=None,
# 	use_fast_tokenizer=False,
# 	padding_side="left",
# ):
# 	if not tokenizer_name_or_path:
# 		tokenizer_name_or_path = model_name_or_path
# 	tokenizer = AutoTokenizer.from_pretrained(
# 		tokenizer_name_or_path,
# 		use_fast=use_fast_tokenizer,
# 		padding_side=padding_side,
# 		trust_remote_code=True,
# 	)

# 	# set pad token to eos token if pad token is not set
# 	if tokenizer.pad_token is None:
# 		if tokenizer.unk_token:
# 			tokenizer.pad_token = tokenizer.unk_token
# 			tokenizer.pad_token_id = tokenizer.unk_token_id
# 		elif tokenizer.eos_token:
# 			tokenizer.pad_token = tokenizer.eos_token
# 			tokenizer.pad_token_id = tokenizer.eos_token_id
# 		else:
# 			raise ValueError(
# 				"You are using a new tokenizer without a pad token."
# 				"This is not supported by this script."
# 			)

# 	return tokenizer


# def _test_generate_completions():
# 	model_name_or_path = "../models/codellama_7b/v1-16k"
# 	llm, tokenizer = load_hf_lm_and_tokenizer(
# 		model_name_or_path=model_name_or_path,
# 		load_in_half=True,
# 		use_fast_tokenizer=True,
# 		use_safetensors=True,
# 	)
# 	# some math word problems
# 	prompts = [
# 		"---\n1+1=2\n---2+2=4\n---3+3=6\n---4+4=8\n---5+5=10\n---6+6=",
# 		"---\n1+1=2\n---12+12=24\n---3+3=6\n---12345+12345=",
# 		# "A train leaves Chicago at 7am and travels at 60mph. Another train leaves Chicago at 9am and travels at 80mph. When will the second train overtake the first?",
# 		# "The sum of two numbers is 10. The difference of the same two numbers is 4. What are the two numbers?",
# 	]

# 	stop_sequences = ["\n\n\n", "---"]
# 	# Because many tokenizers will treat the word after space differently from the original word alone,
# 	# to be consistent, we add a space before tokenization and remove it after tokenization.
# 	# stop_id_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
# 	outputs = generate_completions(
# 		model=llm,
# 		tokenizer=tokenizer,
# 		prompts=prompts,
# 		max_new_tokens=128,
# 		batch_size=16,
# 		# stop_id_sequences=stop_id_sequences,
# 		stop_id_sequences=stop_sequences,
# 	)
# 	print(outputs)


# # @torch.no_grad()
# # def precompute_kv_cache(model, batch_input_ids, attention_mask):
# # 	print("precompute_kv_cache")
# # 	past_key_values = DynamicCache()
# # 	cache_input = {
# # 		"input_ids": batch_input_ids[:, :-1],
# # 		"attention_mask": attention_mask[:, :-1],
# # 	}

# # 	# Compute correct position_ids before caching
# # 	seq_lengths = cache_input["attention_mask"].sum(dim=1)
# # 	position_ids = torch.zeros_like(cache_input["input_ids"])
# # 	for i in range(cache_input["input_ids"].shape[0]):
# # 		valid_len = seq_lengths[i].int()
# # 		position_ids[i, -valid_len:] = torch.arange(valid_len)
# # 	cache_input["position_ids"] = position_ids

# # 	with torch.no_grad():
# # 		model(**cache_input, past_key_values=past_key_values, use_cache=True)

# # 	return past_key_values


# # def steer_kv_cache(cache, steering_kv):
# # 	for layer_idx, past_keys in steering_kv["values"].items():
# # 		sv = (
# # 			steering_kv["values"][layer_idx]
# # 			.clone()
# # 			.to(cache.value_cache[layer_idx].device)
# # 		)
# # 		cache.value_cache[layer_idx][:, :, -1, :] += sv * 1

# # 	for layer_idx, past_keys in steering_kv["keys"].items():
# # 		sv = (
# # 			steering_kv["keys"][layer_idx].clone().to(cache.key_cache[layer_idx].device)
# # 		)
# # 		cache.key_cache[layer_idx][:, :, -1, :] += sv * 1

# # 	return cache
