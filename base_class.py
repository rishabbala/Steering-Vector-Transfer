import gc
import json
import os
import time

import torch
from evaluate import evaluate
from model_utils import (
	generate_completions,
	load_hf_lm_and_tokenizer,
	# load_hf_tokenizer,
	prepare_data,
)
from parser import (
	run_execute,
)
from tqdm import tqdm
from transformers import AutoConfig
from utils import (
	save_jsonl,
)


class ModelDiff:
	def __init__(self):
		return

	def is_multi_choice(self, answer):
		if answer is None:
			return False
		for c in answer:
			if c not in ["A", "B", "C", "D", "E"]:
				return False
		return True

	def create_response_prompt_and_format(
		self, args, data_name, prompt_type, reverse=False, num_shots=0
	):
		samples = prepare_data(data_name, prompt_type, args, reverse, num_shots)
		print("=" * 50)
		print("data:", data_name, " ,remain samples:", len(samples))
		if len(samples) > 0:
			print(samples[0])

		prompts = [sample["prompt"] for sample in samples]
		print("Prompts:", prompts[0])

		stop_words = [
			"</s>",
			"<|im_end|>",
			"<|endoftext|>",
			"assistant",
			"Assistant",
			"user",
			"_end",
			"_start",
			"Question:",
			"Question",
			"Continuation",
			"Continuing",
			"Continues",
			"You are an AI",
			"[Question]",
			# "A:",
			# "B:",
			# "C:",
			# "D:",
			"```",
			"I hope",
			"Subproblem",
			"</atok>",
		]

		return (
			samples,
			prompts,
			stop_words,
		)

	@torch.no_grad()
	def generate_response(
		self,
		args,
		data_name,
		prompt_type,
		out_file,
		apply_kv_cache_diff=False,
		num_shots=0,
	):
		(
			samples,
			prompts,
			stop_words,
		) = self.create_response_prompt_and_format(
			args, data_name, prompt_type, num_shots=num_shots
		)

		executor = None
		end_prompts = []

		start_time = time.time()
		print("-" * 20)
		print(prompts[0])

		if args.ignore_stop_words:
			stop_words = None

		outputs = generate_completions(
			model=self.model,
			tokenizer=self.tokenizer,
			prompts=prompts,
			max_new_tokens=args.max_new_tokens,
			batch_size=args.batch_size,
			stop_id_sequences=stop_words,
			do_sample=args.do_sample,
		)

		assert len(outputs) == len(prompts)
		print("Outputs:", outputs[0])

		for i, (query, output) in enumerate(zip(prompts, outputs)):
			output = output.rstrip()
			query += output
			end_prompts.append((i, query))

		end_prompts = sorted(end_prompts, key=lambda x: x[0])

		# remove input_prompt from end_prompt
		codes = []
		for i in range(len(prompts)):
			_, end_prompt = end_prompts[i]
			code = end_prompt.split(prompts[i])[-1].strip()
			if stop_words is not None:
				for stop_word in stop_words:
					if stop_word in code:
						code = code.split(stop_word)[0].strip()
			codes.append(code)

		print("*" * 20)
		print(codes[0])
		print("*" * 20)

		# extract preds
		results = [
			run_execute(
				executor, codes[c], args.prompt_type, data_name, samples=samples[c]
			)
			for c in range(len(codes))
		]
		time_use = time.time() - start_time

		all_samples = []

		# put results back to examples
		for i, sample in enumerate(samples):
			code = codes[i : (i + 1)]
			result = results[i : (i + 1)]
			preds = (
				[item[0] for item in result]
				if data_name not in ["ifeval", "pop_qa"]
				else None
			)
			reports = [item[1] for item in result]

			sample.update({"code": code, "pred": preds, "report": reports})
			all_samples.append(sample)

		all_samples, result_json = evaluate(
			data_name=data_name,
			prompt_type=args.prompt_type,
			samples=all_samples,
			execute=True,
		)

		# save outputs
		save_jsonl(all_samples, out_file)

		result_json["time_use_in_second"] = time_use
		result_json["time_use_in_minite"] = (
			f"{int(time_use // 60)}:{int(time_use % 60):02d}"
		)

		with open(out_file.replace(".jsonl", "_metrics.json"), "w") as f:
			json.dump(result_json, f, indent=4)

		return result_json

	def load_model_tokenizer(
		self,
		model_name_or_path,
		steering_vector=None,
		transfer=False,
		alpha=1.0,
		device_map="auto",
		dtype=torch.bfloat16,
	):
		self.model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=model_name_or_path,
			use_fast_tokenizer=True,
			steering_vector=steering_vector,
			transfer=transfer,
			alpha=alpha,
			device_map=device_map,
			dtype=dtype,
		)
		self.model = self.model.eval()

	@torch.no_grad()
	def generate_svd_transform(self, args, prompts, pth):
		# LAST HIDDEN STATE
		all_base_hs = []
		all_sub_hs = []

		## Sub Model
		self.sub_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.sub_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.float32,
			use_flash_attn=False,  ## no flash attn for float 32
		)

		self.sub_model = self.sub_model.eval()
		print(f"Loaded Sub Model {args.sub_model}")
		sub_config = AutoConfig.from_pretrained(args.sub_model, trust_remote_code=True)

		total_ex = len(prompts)

		prompt_type = args.prompt_type.split(",")[0]
		if "chat" in prompt_type:
			add_special_tokens = False
		else:
			add_special_tokens = True

		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompts = prompts[i]
			tokenized_prompts = self.tokenizer(
				batch_dataset_prompts,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens,
			)

			batch_input_ids_prompts = tokenized_prompts.input_ids
			attention_mask_prompts = tokenized_prompts.attention_mask

			if self.sub_model.device.type == "cuda":
				batch_input_ids_prompts = batch_input_ids_prompts.to(
					next(self.sub_model.parameters()).device
				)
				attention_mask_prompts = attention_mask_prompts.to(
					next(self.sub_model.parameters()).device
				)

			sub_model_prefill = self.sub_model.model(
				input_ids=batch_input_ids_prompts,
				attention_mask=attention_mask_prompts,
				output_hidden_states=True,
			).hidden_states  ## pass to base model, not the causal lm
			## LAST HIDDEN STATE

			# nLayer+1 * bsz * nLen * hSz becomse nLayer * bsz * hSz
			## (nLayer x bsz x hsz)
			all_sub_hidden_states = torch.stack(
				[vec.to("cpu", copy=False) for vec in sub_model_prefill]
			)[1:, :, -1, :]

			del sub_model_prefill, batch_input_ids_prompts, attention_mask_prompts

			all_sub_hs.append(all_sub_hidden_states)

			del all_sub_hidden_states
			gc.collect()

		del self.sub_model, self.tokenizer
		gc.collect()

		## Base Model
		self.base_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.base_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.float32,
			use_flash_attn=False,
		)

		self.base_model = self.base_model.eval()
		print(f"Loaded Base Model {args.base_model}")

		base_config = AutoConfig.from_pretrained(
			args.base_model, trust_remote_code=True
		)

		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompts = prompts[i]
			tokenized_prompts = self.tokenizer(
				batch_dataset_prompts,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens,
			)

			batch_input_ids_prompts = tokenized_prompts.input_ids
			attention_mask_prompts = tokenized_prompts.attention_mask

			if self.base_model.device.type == "cuda":
				batch_input_ids_prompts = batch_input_ids_prompts.to(
					next(self.base_model.parameters()).device
				)
				attention_mask_prompts = attention_mask_prompts.to(
					next(self.base_model.parameters()).device
				)

			base_model_prefill = self.base_model.model(
				input_ids=batch_input_ids_prompts,
				attention_mask=attention_mask_prompts,
				output_hidden_states=True,
			).hidden_states

			## (nLayer x bsz x hsz)
			all_base_hidden_states = torch.stack(
				[vec.to("cpu", copy=False) for vec in base_model_prefill]
			)[1:, :, -1, :]

			del base_model_prefill, batch_input_ids_prompts, attention_mask_prompts

			all_base_hs.append(all_base_hidden_states)

			del all_base_hidden_states
			gc.collect()

		del self.base_model, self.tokenizer
		gc.collect()

		# Stack layers to get final shape: (num_layers, total_valid_tokens, hidden_size)
		all_sub_hs = torch.cat(all_sub_hs, dim=1)
		all_base_hs = torch.cat(all_base_hs, dim=1)

		if hasattr(sub_config, "text_config"):
			sub_layers = sub_config.text_config.num_hidden_layers
			sub_hidden_size = sub_config.text_config.hidden_size
		else:
			sub_layers = sub_config.num_hidden_layers
			sub_hidden_size = sub_config.hidden_size

		if hasattr(base_config, "text_config"):
			base_layers = base_config.text_config.num_hidden_layers
			base_hidden_size = base_config.text_config.hidden_size
		else:
			base_layers = base_config.num_hidden_layers
			base_hidden_size = base_config.hidden_size

		transform = torch.zeros(
			(
				base_layers,
				sub_hidden_size,
				base_hidden_size,
			)
		)

		for Lb in range(base_layers):
			############ DENSE
			Ls = min(
				((sub_layers - 1) * Lb) // (base_layers - 1),
				sub_layers - 1,
			)  ## linear in

			## num valid tokens x hidden size.
			## As we only use the last token seq_len = 1, so numEx x seq_len = numEx
			## as we have numEx < hsz, with full_matrices=False, we get Vh_sub -> (min(numEx, hsz) x hsz) = (numEx x hsz)
			## V_h -> (numEx x hsz)
			_, _, Vh_sub = torch.linalg.svd(
				all_sub_hs[Ls],
				full_matrices=False,
			)
			_, _, Vh_base = torch.linalg.svd(
				all_base_hs[Lb],
				full_matrices=False,
			)

			## min(First r largest singular vectors, numEx) = k
			## (k x hsz)
			Vh_sub = Vh_sub[: min(Vh_sub.shape[0], args.rank), :]
			Vh_base = Vh_base[: min(Vh_base.shape[0], args.rank), :]

			## Project the original hidden state onto the low dimension space
			## (numEx x hsz(s/b)) @ (hsz(s/b) x k) -> (numEx x k)
			proj_sub = all_sub_hs[Ls] @ Vh_sub.T
			proj_base = all_base_hs[Lb] @ Vh_base.T

			## (k x numEx) @ (numEx x k) -> (k x k)
			tf = torch.linalg.pinv(proj_sub)
			tf = tf @ proj_base

			## Compute final transformation as sub space -> sub low proj space -> base low proj space -> base space
			## (hsz_sub x k) @ (k x k) @ (k x hsz_base) -> (hsz_sub x hsz_base)
			tf = Vh_sub.T @ tf @ Vh_base

			print(tf.shape, Vh_base.shape, Vh_sub.shape)
			tf = tf.cpu()
			transform[Lb] = tf

		print("Saving HS Diff", pth)
		torch.save(transform, pth)

	@torch.no_grad()
	def generate_svd_transform_zero_centered(self, args, prompts, pth):
		# LAST HIDDEN STATE
		all_base_hs = []
		all_sub_hs = []

		## Sub Model
		self.sub_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.sub_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.float32,
			use_flash_attn=False,  ## no flash attn for float 32
		)

		self.sub_model = self.sub_model.eval()
		print(f"Loaded Sub Model {args.sub_model}")

		sub_config = AutoConfig.from_pretrained(args.sub_model, trust_remote_code=True)

		total_ex = len(prompts)

		prompt_type = args.prompt_type.split(",")[0]
		if "chat" in prompt_type:
			add_special_tokens = False
		else:
			add_special_tokens = True

		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompts = prompts[i]
			tokenized_prompts = self.tokenizer(
				batch_dataset_prompts,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens,
			)

			batch_input_ids_prompts = tokenized_prompts.input_ids
			attention_mask_prompts = tokenized_prompts.attention_mask

			if self.sub_model.device.type == "cuda":
				batch_input_ids_prompts = batch_input_ids_prompts.to(
					next(self.sub_model.parameters()).device
				)
				attention_mask_prompts = attention_mask_prompts.to(
					next(self.sub_model.parameters()).device
				)

			sub_model_prefill = self.sub_model.model(
				input_ids=batch_input_ids_prompts,
				attention_mask=attention_mask_prompts,
				output_hidden_states=True,
			).hidden_states  ## pass to base model, not the causal lm
			## LAST HIDDEN STATE

			# nLayer+1 * bsz * nLen * hSz becomse nLayer * bsz * hSz
			## (nLayer x bsz x hsz)
			all_sub_hidden_states = torch.stack(
				[vec.to("cpu", copy=False) for vec in sub_model_prefill]
			)[1:, :, -1, :]

			del sub_model_prefill, batch_input_ids_prompts, attention_mask_prompts

			all_sub_hs.append(all_sub_hidden_states)

			del all_sub_hidden_states
			gc.collect()

		del self.sub_model, self.tokenizer
		gc.collect()

		## Base Model
		self.base_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.base_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.float32,
			use_flash_attn=False,
		)

		self.base_model = self.base_model.eval()
		print(f"Loaded Base Model {args.base_model}")

		base_config = AutoConfig.from_pretrained(
			args.base_model, trust_remote_code=True
		)

		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompts = prompts[i]
			tokenized_prompts = self.tokenizer(
				batch_dataset_prompts,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens,
			)

			batch_input_ids_prompts = tokenized_prompts.input_ids
			attention_mask_prompts = tokenized_prompts.attention_mask

			if self.base_model.device.type == "cuda":
				batch_input_ids_prompts = batch_input_ids_prompts.to(
					next(self.base_model.parameters()).device
				)
				attention_mask_prompts = attention_mask_prompts.to(
					next(self.base_model.parameters()).device
				)

			base_model_prefill = self.base_model.model(
				input_ids=batch_input_ids_prompts,
				attention_mask=attention_mask_prompts,
				output_hidden_states=True,
			).hidden_states

			## (nLayer x bsz x hsz)
			all_base_hidden_states = torch.stack(
				[vec.to("cpu", copy=False) for vec in base_model_prefill]
			)[1:, :, -1, :]

			del base_model_prefill, batch_input_ids_prompts, attention_mask_prompts

			all_base_hs.append(all_base_hidden_states)

			del all_base_hidden_states
			gc.collect()

		del self.base_model, self.tokenizer
		gc.collect()

		# Stack layers to get final shape: (num_layers, total_valid_tokens, hidden_size)
		all_sub_hs = torch.cat(all_sub_hs, dim=1)
		all_base_hs = torch.cat(all_base_hs, dim=1)

		if hasattr(sub_config, "text_config"):
			sub_layers = sub_config.text_config.num_hidden_layers
			sub_hidden_size = sub_config.text_config.hidden_size
		else:
			sub_layers = sub_config.num_hidden_layers
			sub_hidden_size = sub_config.hidden_size

		if hasattr(base_config, "text_config"):
			base_layers = base_config.text_config.num_hidden_layers
			base_hidden_size = base_config.text_config.hidden_size
		else:
			base_layers = base_config.num_hidden_layers
			base_hidden_size = base_config.hidden_size

		transform = torch.zeros(
			(
				base_layers,
				sub_hidden_size,
				base_hidden_size,
			)
		)

		for Lb in range(base_layers):
			############ DENSE
			Ls = min(
				((sub_layers - 1) * Lb) // (base_layers - 1),
				sub_layers - 1,
			)  ## linear in

			## num valid tokens x hidden size.
			## As we only use the last token seq_len = 1, so numEx x seq_len = numEx
			## as we have numEx < hsz, with full_matrices=False, we get Vh_sub -> (min(numEx, hsz) x hsz) = (numEx x hsz)
			## V_h -> (numEx x hsz)

			## Zero-centered
			all_sub_hs[Ls] = all_sub_hs[Ls] - all_sub_hs[Ls].mean(dim=0)
			all_base_hs[Lb] = all_base_hs[Lb] - all_base_hs[Lb].mean(dim=0)

			_, _, Vh_sub = torch.linalg.svd(
				all_sub_hs[Ls],
				full_matrices=False,
			)
			_, _, Vh_base = torch.linalg.svd(
				all_base_hs[Lb],
				full_matrices=False,
			)

			## min(First r largest singular vectors, numEx) = k
			## (k x hsz)
			Vh_sub = Vh_sub[: min(Vh_sub.shape[0], args.rank), :]
			Vh_base = Vh_base[: min(Vh_base.shape[0], args.rank), :]

			## Project the original hidden state onto the low dimension space
			## (numEx x hsz(s/b)) @ (hsz(s/b) x k) -> (numEx x k)
			proj_sub = all_sub_hs[Ls] @ Vh_sub.T
			proj_base = all_base_hs[Lb] @ Vh_base.T

			## (k x numEx) @ (numEx x k) -> (k x k)
			tf = torch.linalg.pinv(proj_sub)
			tf = tf @ proj_base

			## Compute final transformation as sub space -> sub low proj space -> base low proj space -> base space
			## (hsz_sub x k) @ (k x k) @ (k x hsz_base) -> (hsz_sub x hsz_base)
			tf = Vh_sub.T @ tf @ Vh_base

			print(tf.shape, Vh_base.shape, Vh_sub.shape)
			tf = tf.cpu()
			transform[Lb] = tf

		print("Saving HS Diff", pth)
		torch.save(transform, pth)

	@torch.no_grad()
	def generate_hs_diff(self, args, dataset_pair, pth):
		sub_config = AutoConfig.from_pretrained(args.sub_model, trust_remote_code=True)

		prompt_types = args.prompt_type.split(",")

		if hasattr(sub_config, "text_config"):
			sub_layers = sub_config.text_config.num_hidden_layers
			sub_hidden_size = sub_config.text_config.hidden_size
		else:
			sub_layers = sub_config.num_hidden_layers
			sub_hidden_size = sub_config.hidden_size

		steering_vector = torch.zeros((sub_layers, sub_hidden_size))
		total_ex = len(dataset_pair[0])

		self.add_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.add_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.bfloat16,
			use_flash_attn=True,
		)
		self.add_model = self.add_model.eval()

		if "chat" in prompt_types[0]:
			add_special_tokens_0 = False
		else:
			add_special_tokens_0 = True
		if "chat" in prompt_types[1]:
			add_special_tokens_1 = False
		else:
			add_special_tokens_1 = True

		all_prompts_0 = []
		all_prompts_1 = []
		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompt_0 = dataset_pair[0][i]["prompt"]
			batch_dataset_prompt_1 = dataset_pair[1][i]["prompt"]
			tokenized_dataset_prompt_0 = self.tokenizer(
				batch_dataset_prompt_0,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens_0,
			)
			tokenized_dataset_prompt_1 = self.tokenizer(
				batch_dataset_prompt_1,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens_1,
			)

			## pad to same length
			max_len = max(
				tokenized_dataset_prompt_0.input_ids.shape[1],
				tokenized_dataset_prompt_1.input_ids.shape[1],
			)
			tokenized_dataset_prompt_0 = self.tokenizer(
				batch_dataset_prompt_0,
				padding="max_length",
				max_length=max_len,
				return_tensors="pt",
				add_special_tokens=add_special_tokens_0,
			)
			tokenized_dataset_prompt_1 = self.tokenizer(
				batch_dataset_prompt_1,
				padding="max_length",
				max_length=max_len,
				return_tensors="pt",
				add_special_tokens=add_special_tokens_1,
			)

			all_prompts_0.append(tokenized_dataset_prompt_0)
			all_prompts_1.append(tokenized_dataset_prompt_1)

			del (
				batch_dataset_prompt_0,
				batch_dataset_prompt_1,
				tokenized_dataset_prompt_0,
				tokenized_dataset_prompt_1,
			)

		all_add_model_hidden_states = []
		for i in tqdm(range(0, total_ex)):
			tokenized_dataset_prompt_1 = all_prompts_1[i]
			batch_input_ids_prompt_1 = tokenized_dataset_prompt_1.input_ids
			attention_mask_prompt_1 = tokenized_dataset_prompt_1.attention_mask

			if self.add_model.device.type == "cuda":
				batch_input_ids_prompt_1 = batch_input_ids_prompt_1.to(
					next(self.add_model.parameters()).device
				)
				attention_mask_prompt_1 = attention_mask_prompt_1.to(
					next(self.add_model.parameters()).device
				)

			add_model_prefill = self.add_model.model(
				input_ids=batch_input_ids_prompt_1,
				attention_mask=attention_mask_prompt_1,
				output_hidden_states=True,
			)

			del batch_input_ids_prompt_1, attention_mask_prompt_1

			add_model_hidden_states = add_model_prefill.hidden_states
			add_model_hidden_states = torch.stack(
				[vec.detach().cpu() for vec in add_model_hidden_states]
			)

			## LAST HIDDEN STATE
			## Answer: Let's think step by step.\n
			## Take the difference in hidden states at '\n'

			# # nLayer+1 x bsz x nLen x hSz becomse nLayer x bsz x hSz
			add_model_last_hidden_states = add_model_hidden_states[1:, :, -1, :]

			print(add_model_last_hidden_states.shape)
			all_add_model_hidden_states.append(add_model_last_hidden_states)

			del add_model_last_hidden_states, add_model_hidden_states, add_model_prefill
			gc.collect()

		del self.add_model, self.tokenizer
		gc.collect()

		self.sub_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.sub_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.bfloat16,
			use_flash_attn=False,
		)
		self.sub_model = self.sub_model.eval()

		for i in tqdm(range(0, total_ex)):
			tokenized_dataset_prompt_0 = all_prompts_0[i]
			batch_input_ids_prompt_0 = tokenized_dataset_prompt_0.input_ids
			attention_mask_prompt_0 = tokenized_dataset_prompt_0.attention_mask

			if self.sub_model.device.type == "cuda":
				batch_input_ids_prompt_0 = batch_input_ids_prompt_0.to(
					next(self.sub_model.parameters()).device
				)
				attention_mask_prompt_0 = attention_mask_prompt_0.to(
					next(self.sub_model.parameters()).device
				)

			sub_model_prefill = self.sub_model.model(
				input_ids=batch_input_ids_prompt_0,
				attention_mask=attention_mask_prompt_0,
				output_hidden_states=True,
			)

			del batch_input_ids_prompt_0, attention_mask_prompt_0

			sub_model_hidden_states = sub_model_prefill.hidden_states
			sub_model_hidden_states = torch.stack(
				[vec.detach().cpu() for vec in sub_model_hidden_states]
			)

			## LAST HIDDEN STATE
			## Answer: Let's think step by step.\n
			## Take the difference in hidden states at '\n'

			# # nLayer+1 x bsz x nLen x hSz becomse nLayer x bsz x hSz
			sub_model_last_hidden_states = sub_model_hidden_states[1:, :, -1, :]

			print(sub_model_last_hidden_states.shape)
			hs_diff = all_add_model_hidden_states[i] - sub_model_last_hidden_states
			## becomes nLayer x hSz
			steering_vector += torch.sum(
				hs_diff / total_ex,
				dim=1,
			)

			del sub_model_last_hidden_states, sub_model_hidden_states, sub_model_prefill
			gc.collect()

		print(
			"Saving HS Diff",
			pth,
		)

		print("Saving HS Diff", pth)
		torch.save(steering_vector, pth)

	@torch.no_grad()
	def generate_hs_pca_post_diff(self, args, dataset_pair, pth):
		sub_config = AutoConfig.from_pretrained(args.sub_model, trust_remote_code=True)

		prompt_types = args.prompt_type.split(",")

		if hasattr(sub_config, "text_config"):
			sub_layers = sub_config.text_config.num_hidden_layers
			sub_hidden_size = sub_config.text_config.hidden_size
		else:
			sub_layers = sub_config.num_hidden_layers
			sub_hidden_size = sub_config.hidden_size

		steering_vector = torch.zeros((sub_layers, sub_hidden_size))
		all_hs_diffs = []
		total_ex = len(dataset_pair[0])

		self.add_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.add_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.float32,
			use_flash_attn=False,
		)
		self.add_model = self.add_model.eval()

		if "chat" in prompt_types[0]:
			add_special_tokens_0 = False
		else:
			add_special_tokens_0 = True
		if "chat" in prompt_types[1]:
			add_special_tokens_1 = False
		else:
			add_special_tokens_1 = True

		all_prompts_0 = []
		all_prompts_1 = []
		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompt_0 = dataset_pair[0][i]["prompt"]
			batch_dataset_prompt_1 = dataset_pair[1][i]["prompt"]
			tokenized_dataset_prompt_0 = self.tokenizer(
				batch_dataset_prompt_0,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens_0,
			)
			tokenized_dataset_prompt_1 = self.tokenizer(
				batch_dataset_prompt_1,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens_1,
			)

			## pad to same length
			max_len = max(
				tokenized_dataset_prompt_0.input_ids.shape[1],
				tokenized_dataset_prompt_1.input_ids.shape[1],
			)
			tokenized_dataset_prompt_0 = self.tokenizer(
				batch_dataset_prompt_0,
				padding="max_length",
				max_length=max_len,
				return_tensors="pt",
				add_special_tokens=add_special_tokens_0,
			)
			tokenized_dataset_prompt_1 = self.tokenizer(
				batch_dataset_prompt_1,
				padding="max_length",
				max_length=max_len,
				return_tensors="pt",
				add_special_tokens=add_special_tokens_1,
			)

			all_prompts_0.append(tokenized_dataset_prompt_0)
			all_prompts_1.append(tokenized_dataset_prompt_1)

			del (
				batch_dataset_prompt_0,
				batch_dataset_prompt_1,
				tokenized_dataset_prompt_0,
				tokenized_dataset_prompt_1,
			)

		all_add_model_hidden_states = []
		for i in tqdm(range(0, total_ex)):
			tokenized_dataset_prompt_1 = all_prompts_1[i]
			batch_input_ids_prompt_1 = tokenized_dataset_prompt_1.input_ids
			attention_mask_prompt_1 = tokenized_dataset_prompt_1.attention_mask

			if self.add_model.device.type == "cuda":
				batch_input_ids_prompt_1 = batch_input_ids_prompt_1.to(
					next(self.add_model.parameters()).device
				)
				attention_mask_prompt_1 = attention_mask_prompt_1.to(
					next(self.add_model.parameters()).device
				)

			add_model_prefill = self.add_model.model(
				input_ids=batch_input_ids_prompt_1,
				attention_mask=attention_mask_prompt_1,
				output_hidden_states=True,
			)

			del batch_input_ids_prompt_1, attention_mask_prompt_1

			add_model_hidden_states = add_model_prefill.hidden_states
			add_model_hidden_states = torch.stack(
				[vec.detach().cpu() for vec in add_model_hidden_states]
			)

			## LAST HIDDEN STATE
			## Answer: Let's think step by step.\n
			## Take the difference in hidden states at '\n'

			# # nLayer+1 x bsz x nLen x hSz becomse nLayer x bsz x hSz
			add_model_last_hidden_states = add_model_hidden_states[1:, :, -1, :]

			print(add_model_last_hidden_states.shape)
			all_add_model_hidden_states.append(add_model_last_hidden_states)

			del add_model_last_hidden_states, add_model_hidden_states, add_model_prefill
			gc.collect()

		del self.add_model, self.tokenizer
		gc.collect()

		self.sub_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.sub_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.float32,
			use_flash_attn=False,
		)
		self.sub_model = self.sub_model.eval()

		for i in tqdm(range(0, total_ex)):
			tokenized_dataset_prompt_0 = all_prompts_0[i]
			batch_input_ids_prompt_0 = tokenized_dataset_prompt_0.input_ids
			attention_mask_prompt_0 = tokenized_dataset_prompt_0.attention_mask

			if self.sub_model.device.type == "cuda":
				batch_input_ids_prompt_0 = batch_input_ids_prompt_0.to(
					next(self.sub_model.parameters()).device
				)
				attention_mask_prompt_0 = attention_mask_prompt_0.to(
					next(self.sub_model.parameters()).device
				)

			sub_model_prefill = self.sub_model.model(
				input_ids=batch_input_ids_prompt_0,
				attention_mask=attention_mask_prompt_0,
				output_hidden_states=True,
			)

			del batch_input_ids_prompt_0, attention_mask_prompt_0

			sub_model_hidden_states = sub_model_prefill.hidden_states
			sub_model_hidden_states = torch.stack(
				[vec.detach().cpu() for vec in sub_model_hidden_states]
			)

			## LAST HIDDEN STATE
			## Answer: Let's think step by step.\n
			## Take the difference in hidden states at '\n'

			# # nLayer+1 x bsz x nLen x hSz becomse nLayer x bsz x hSz
			sub_model_last_hidden_states = sub_model_hidden_states[1:, :, -1, :]

			print(sub_model_last_hidden_states.shape)
			hs_diff = all_add_model_hidden_states[i] - sub_model_last_hidden_states
			## becomes nLayer x hSz
			all_hs_diffs.append(hs_diff.squeeze(1))

			del sub_model_last_hidden_states, sub_model_hidden_states, sub_model_prefill
			gc.collect()

		all_hs_diffs = torch.stack(all_hs_diffs)

		for layer in range(sub_layers):
			layer_diffs = all_hs_diffs[:, layer, :]

			layer_diffs = layer_diffs - layer_diffs.mean(dim=0)
			_, _, Vh_sub = torch.linalg.svd(
				layer_diffs,
				full_matrices=False,
			)
			Vh_sub = Vh_sub[0, :]

			steering_vector[layer] = Vh_sub

		print("Saving HS Diff", pth)
		torch.save(steering_vector, pth)

	@torch.no_grad()
	def generate_all_vectors(self, args, dataset_pair, pth):
		prompt_types = args.prompt_type.split(",")

		total_ex = len(dataset_pair[0])

		self.base_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.base_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.bfloat16,
			use_flash_attn=True,
		)
		self.base_model = self.base_model.eval()

		if "chat" in prompt_types[0]:
			add_special_tokens_0 = False
		else:
			add_special_tokens_0 = True
		if "chat" in prompt_types[1]:
			add_special_tokens_1 = False
		else:
			add_special_tokens_1 = True

		all_prompts_2 = []
		base_model_vectors = []
		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompt_2 = dataset_pair[0][i]["prompt"]
			tokenized_dataset_prompt_2 = self.tokenizer(
				batch_dataset_prompt_2,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens_0,
			)

			all_prompts_2.append(tokenized_dataset_prompt_2)

			del (
				batch_dataset_prompt_2,
				tokenized_dataset_prompt_2,
			)

		for i in tqdm(range(0, total_ex)):
			tokenized_dataset_prompt_2 = all_prompts_2[i]
			batch_input_ids_prompt_2 = tokenized_dataset_prompt_2.input_ids
			attention_mask_prompt_2 = tokenized_dataset_prompt_2.attention_mask

			if self.base_model.device.type == "cuda":
				batch_input_ids_prompt_2 = batch_input_ids_prompt_2.to(
					next(self.base_model.parameters()).device
				)
				attention_mask_prompt_2 = attention_mask_prompt_2.to(
					next(self.base_model.parameters()).device
				)

			base_model_prefill = self.base_model.model(
				input_ids=batch_input_ids_prompt_2,
				attention_mask=attention_mask_prompt_2,
				output_hidden_states=True,
			)

			del batch_input_ids_prompt_2, attention_mask_prompt_2

			base_model_hidden_states = base_model_prefill.hidden_states
			base_model_hidden_states = torch.stack(
				[vec.detach().cpu() for vec in base_model_hidden_states]
			)

			## LAST HIDDEN STATE
			## Answer: Let's think step by step.\n
			## Take the difference in hidden states at '\n'

			# # nLayer+1 x bsz x nLen x hSz becomse nLayer x bsz x hSz
			base_model_last_hidden_states = base_model_hidden_states[1:, :, -1, :]
			base_model_vectors.append(base_model_last_hidden_states)

			del (
				base_model_last_hidden_states,
				base_model_hidden_states,
				base_model_prefill,
			)
			gc.collect()

		base_model_vectors = torch.stack(base_model_vectors, dim=1)
		print(
			"Saving Base Model Vectors",
			f"/projects/llms-lab/transfer_compare/spectral_entropy/{args.base_model.replace('/', '_')}_{args.data_name}_{args.split}_{prompt_types[0]}.pth",
		)
		torch.save(
			base_model_vectors,
			f"/projects/llms-lab/transfer_compare/spectral_entropy/{args.base_model.replace('/', '_')}_{args.data_name}_{args.split}_{prompt_types[0]}.pth",
		)

		del self.base_model, self.tokenizer
		gc.collect()

		add_model_vectors = []
		sub_model_vectors = []

		self.add_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.add_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.bfloat16,
			use_flash_attn=True,
		)
		self.add_model = self.add_model.eval()

		if "chat" in prompt_types[0]:
			add_special_tokens_0 = False
		else:
			add_special_tokens_0 = True
		if "chat" in prompt_types[1]:
			add_special_tokens_1 = False
		else:
			add_special_tokens_1 = True

		all_prompts_0 = []
		all_prompts_1 = []
		for i in tqdm(range(0, total_ex)):
			batch_dataset_prompt_0 = dataset_pair[0][i]["prompt"]
			batch_dataset_prompt_1 = dataset_pair[1][i]["prompt"]
			tokenized_dataset_prompt_0 = self.tokenizer(
				batch_dataset_prompt_0,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens_0,
			)
			tokenized_dataset_prompt_1 = self.tokenizer(
				batch_dataset_prompt_1,
				padding="longest",
				return_tensors="pt",
				add_special_tokens=add_special_tokens_1,
			)

			## pad to same length
			max_len = max(
				tokenized_dataset_prompt_0.input_ids.shape[1],
				tokenized_dataset_prompt_1.input_ids.shape[1],
			)
			tokenized_dataset_prompt_0 = self.tokenizer(
				batch_dataset_prompt_0,
				padding="max_length",
				max_length=max_len,
				return_tensors="pt",
				add_special_tokens=add_special_tokens_0,
			)
			tokenized_dataset_prompt_1 = self.tokenizer(
				batch_dataset_prompt_1,
				padding="max_length",
				max_length=max_len,
				return_tensors="pt",
				add_special_tokens=add_special_tokens_1,
			)

			all_prompts_0.append(tokenized_dataset_prompt_0)
			all_prompts_1.append(tokenized_dataset_prompt_1)

			del (
				batch_dataset_prompt_0,
				batch_dataset_prompt_1,
				tokenized_dataset_prompt_0,
				tokenized_dataset_prompt_1,
			)

		for i in tqdm(range(0, total_ex)):
			tokenized_dataset_prompt_1 = all_prompts_1[i]
			batch_input_ids_prompt_1 = tokenized_dataset_prompt_1.input_ids
			attention_mask_prompt_1 = tokenized_dataset_prompt_1.attention_mask

			if self.add_model.device.type == "cuda":
				batch_input_ids_prompt_1 = batch_input_ids_prompt_1.to(
					next(self.add_model.parameters()).device
				)
				attention_mask_prompt_1 = attention_mask_prompt_1.to(
					next(self.add_model.parameters()).device
				)

			add_model_prefill = self.add_model.model(
				input_ids=batch_input_ids_prompt_1,
				attention_mask=attention_mask_prompt_1,
				output_hidden_states=True,
			)

			del batch_input_ids_prompt_1, attention_mask_prompt_1

			add_model_hidden_states = add_model_prefill.hidden_states
			add_model_hidden_states = torch.stack(
				[vec.detach().cpu() for vec in add_model_hidden_states]
			)

			## LAST HIDDEN STATE
			## Answer: Let's think step by step.\n
			## Take the difference in hidden states at '\n'

			# # nLayer+1 x bsz x nLen x hSz becomse nLayer x bsz x hSz
			add_model_last_hidden_states = add_model_hidden_states[1:, :, -1, :]
			add_model_vectors.append(add_model_last_hidden_states)

			del add_model_last_hidden_states, add_model_hidden_states, add_model_prefill
			gc.collect()

		add_model_vectors = torch.stack(add_model_vectors, dim=1)
		print(
			"Saving Add Model Vectors",
			f"/projects/llms-lab/transfer_compare/spectral_entropy/{args.add_model.replace('/', '_')}_{args.data_name}_{args.split}_{prompt_types[1]}.pth",
		)
		torch.save(
			add_model_vectors,
			f"/projects/llms-lab/transfer_compare/spectral_entropy/{args.add_model.replace('/', '_')}_{args.data_name}_{args.split}_{prompt_types[1]}.pth",
		)

		del self.add_model, self.tokenizer
		gc.collect()

		self.sub_model, self.tokenizer = load_hf_lm_and_tokenizer(
			model_name_or_path=args.sub_model,
			use_fast_tokenizer=True,
			device_map="auto",
			dtype=torch.bfloat16,
			use_flash_attn=False,
		)
		self.sub_model = self.sub_model.eval()

		for i in tqdm(range(0, total_ex)):
			tokenized_dataset_prompt_0 = all_prompts_0[i]
			batch_input_ids_prompt_0 = tokenized_dataset_prompt_0.input_ids
			attention_mask_prompt_0 = tokenized_dataset_prompt_0.attention_mask

			if self.sub_model.device.type == "cuda":
				batch_input_ids_prompt_0 = batch_input_ids_prompt_0.to(
					next(self.sub_model.parameters()).device
				)
				attention_mask_prompt_0 = attention_mask_prompt_0.to(
					next(self.sub_model.parameters()).device
				)

			sub_model_prefill = self.sub_model.model(
				input_ids=batch_input_ids_prompt_0,
				attention_mask=attention_mask_prompt_0,
				output_hidden_states=True,
			)

			del batch_input_ids_prompt_0, attention_mask_prompt_0

			sub_model_hidden_states = sub_model_prefill.hidden_states
			sub_model_hidden_states = torch.stack(
				[vec.detach().cpu() for vec in sub_model_hidden_states]
			)

			## LAST HIDDEN STATE
			## Answer: Let's think step by step.\n
			## Take the difference in hidden states at '\n'

			# # nLayer+1 x bsz x nLen x hSz becomse nLayer x bsz x hSz
			sub_model_last_hidden_states = sub_model_hidden_states[1:, :, -1, :]
			sub_model_vectors.append(sub_model_last_hidden_states)

			del sub_model_last_hidden_states, sub_model_hidden_states, sub_model_prefill
			gc.collect()

		sub_model_vectors = torch.stack(sub_model_vectors, dim=1)

		print(
			"Saving Sub Model Vectors",
			f"/projects/llms-lab/transfer_compare/spectral_entropy/{args.sub_model.replace('/', '_')}_{args.data_name}_{args.split}_{prompt_types[0]}.pth",
		)
		torch.save(
			sub_model_vectors,
			f"/projects/llms-lab/transfer_compare/spectral_entropy/{args.sub_model.replace('/', '_')}_{args.data_name}_{args.split}_{prompt_types[0]}.pth",
		)
