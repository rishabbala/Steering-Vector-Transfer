# Modified from https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py
import os

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DataLoaderConfiguration
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	get_linear_schedule_with_warmup,
	set_seed,
)

MAX_GPU_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 32
CHECKPOINTING_STEPS = 10  ## STEPS AFTER WHICH TO SAVE
MODEL_NAME = "meta-llama/Llama-3.1-8B"
GRADIENT_ACCUMULATION_STEPS = 2
OUTPUT_DIR = "/projects/llms-lab/train_partial/llama-3.1-8b-gsm8k/"
LEARNING_RATE = 2e-5
EPOCHS = 3
SPEED = 42


def training_function():
	# Initialize accelerator
	dataloader_config = DataLoaderConfiguration()
	# ## LATER ON FOR ADDING TO WANDB
	# if args.with_tracking:
	# 	accelerator = Accelerator(
	# 		cpu=args.cpu,
	# 		mixed_precision=args.mixed_precision,
	# 		dataloader_config=dataloader_config,
	# 		log_with="all",
	# 		project_dir=args.project_dir,
	# 	)
	# else:
	accelerator = Accelerator(
		# cpu=args.cpu,
		# mixed_precision=args.mixed_precision,
		mixed_precision="bf16",
		dataloader_config=dataloader_config,
		deepspeed_plugin=DeepSpeedPlugin(
			gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
			zero_stage=3,
			zero3_init_flag=True,
			zero3_save_16bit_model=True,
			offload_optimizer_device="cpu",
		),
	)

	checkpointing_steps = CHECKPOINTING_STEPS
	# Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
	lr = LEARNING_RATE
	num_epochs = EPOCHS
	seed = SPEED
	batch_size = MAX_GPU_BATCH_SIZE

	# ## LATER ON FOR ADDING TO WANDB
	# # We need to initialize the trackers we use, and also store our configuration
	# if args.with_tracking:
	# 	run = os.path.split(__file__)[-1].split(".")[0]
	# 	accelerator.init_trackers(run, config)

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	tokenizer.padding_side = "left"
	tokenizer.truncation_side = "left"
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	train_dataset = load_dataset("openai/gsm8k", "main", split="train").select(
		range(100)
	)
	eval_dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(100))
	# metric = evaluate.load("glue", "mrpc") # HAVE TO IMPLEMENT CROSS ENTROPY

	def promptFormatter(examples, input_name="question", output_name="answer"):
		# Build separate prompt/target fields from question/answer
		prompts, targets = [], []
		for q, a in zip(examples[input_name], examples[output_name]):
			prompts.append(
				f"#### Question: {q}\nLet's think step by step.\n\n#### Answer:"
			)
			targets.append(f" {a}")
		return {"prompt": prompts, "target": targets}

	## VERIFY MAX LENGTHS
	def tokenize_function(
		examples,
		prompt_name="prompt",
		target_name="target",
		add_eos=True,
		max_length=128,
	):
		# Tokenize prompt and target separately; mask loss on prompt tokens, keep loss on target tokens only
		p_ids = tokenizer(examples[prompt_name], add_special_tokens=False)["input_ids"]
		t_ids = tokenizer(examples[target_name], add_special_tokens=False)["input_ids"]

		input_ids, attention_mask, labels = [], [], []
		eos_id = tokenizer.eos_token_id

		for p, t in zip(p_ids, t_ids):
			seq = p + t + ([eos_id] if add_eos and eos_id is not None else [])
			lab = (
				([-100] * len(p))
				+ t
				+ ([eos_id] if add_eos and eos_id is not None else [])
			)
			mask = (
				[1] * len(seq)
			)  ## setting mask to 0 means its considered padding. So token completely ignored during generatio and loss comp

			if len(seq) > max_length:
				seq = seq[:max_length]
				mask = mask[:max_length]
				lab = lab[:max_length]

			input_ids.append(seq)
			attention_mask.append(mask)
			labels.append(lab)

		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": labels,
		}

	# ## Need it for evaluating model accuracy, not for now
	# def final_answer(examples, field="answer"):
	# 	# Extract gold label after '####' for optional eval post-processing
	# 	labels = [s.split("####")[-1].strip() for s in examples[field]]
	# 	return {"label": labels}

	# Apply the method we just defined to all the examples in all the splits of the dataset
	# starting with the main process first: Why main process first? Convention?
	with accelerator.main_process_first():
		# train_dataset = train_dataset.map(
		# 	final_answer,
		# 	batched=True,
		# 	fn_kwargs={"field": "answer"},
		# 	load_from_cache_file=False,
		# )
		train_dataset = train_dataset.map(
			promptFormatter,
			batched=True,
			fn_kwargs={"input_name": "question", "output_name": "answer"},
			remove_columns=train_dataset.column_names,
			load_from_cache_file=False,
		)
		train_dataset = train_dataset.map(
			tokenize_function,
			batched=True,
			fn_kwargs={
				"prompt_name": "prompt",
				"target_name": "target",
				"max_length": 128,
			},
			load_from_cache_file=False,
		)

		# eval_dataset = eval_dataset.map(
		# 	final_answer,
		# 	batched=True,
		# 	fn_kwargs={"field": "answer"},
		# 	load_from_cache_file=False,
		# )
		eval_dataset = eval_dataset.map(
			promptFormatter,
			batched=True,
			fn_kwargs={"input_name": "question", "output_name": "answer"},
			remove_columns=eval_dataset.column_names,
			load_from_cache_file=False,
		)
		eval_dataset = eval_dataset.map(
			tokenize_function,
			batched=True,
			fn_kwargs={
				"prompt_name": "prompt",
				"target_name": "target",
				"max_length": 128,
			},
			load_from_cache_file=False,
		)

	# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
	# transformers library
	# eval_dataset = eval_dataset.rename_column("answer", "labels")

	# # If the batch size is too big we use gradient accumulation
	# if (
	# 	batch_size > MAX_GPU_BATCH_SIZE
	# 	and accelerator.distributed_type != DistributedType.XLA
	# ):
	# 	gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
	# 	batch_size = MAX_GPU_BATCH_SIZE

	## VERIFY MAX LENGTHS
	def collate_fn(examples):
		# Pad input_ids/attention_mask with tokenizer.pad
		batch_input = {
			"input_ids": [e["input_ids"] for e in examples],
			"attention_mask": [e["attention_mask"] for e in examples],
		}
		padded = tokenizer.pad(
			batch_input, padding="longest", max_length=128, return_tensors="pt"
		)
		max_len = padded["input_ids"].size(1)
		labels = [e["labels"] for e in examples]
		padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
		for i, lab in enumerate(labels):
			padded_labels[i, -min(len(lab), max_len) :] = torch.tensor(
				lab[-min(len(lab), max_len) :], dtype=torch.long
			)
			# padded_labels[i, : min(len(lab), max_len)] = torch.tensor(
			# 	lab[: min(len(lab), max_len)], dtype=torch.long
			# )

		padded["labels"] = padded_labels

		return padded

	# Instantiate dataloaders.
	train_dataloader = DataLoader(
		train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
	)
	eval_dataloader = DataLoader(
		eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
	)

	set_seed(seed)

	model = AutoModelForCausalLM.from_pretrained(
		MODEL_NAME, return_dict=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
	)
	# model = model.to(accelerator.device)

	optimizer = AdamW(params=model.parameters(), lr=lr)

	lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=1,
		num_training_steps=10,
	)

	model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
		accelerator.prepare(
			model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
		)
	)

	overall_step = 0
	starting_epoch = 0

	# ## SKIP FOR NOW
	# # Potentially load in the weights and states from a previous save
	# if args.resume_from_checkpoint:
	# 	if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
	# 		accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
	# 		accelerator.load_state(args.resume_from_checkpoint)
	# 		path = os.path.basename(args.resume_from_checkpoint)
	# 	else:
	# 		# Get the most recent checkpoint
	# 		dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
	# 		dirs.sort(key=os.path.getctime)
	# 		path = dirs[
	# 			-1
	# 		]  # Sorts folders by date modified, most recent checkpoint is the last
	# 	# Extract `epoch_{i}` or `step_{i}`
	# 	training_difference = os.path.splitext(path)[0]

	# 	if "epoch" in training_difference:
	# 		starting_epoch = int(training_difference.replace("epoch_", "")) + 1
	# 		resume_step = None
	# 	else:
	# 		resume_step = int(training_difference.replace("step_", ""))
	# 		starting_epoch = resume_step // len(train_dataloader)
	# 		resume_step -= starting_epoch * len(train_dataloader)

	# Now we train the model
	for epoch in range(starting_epoch, num_epochs):
		model.train()
		total_loss = 0
		active_dataloader = train_dataloader
		for step, batch in enumerate(active_dataloader):
			batch.to(accelerator.device)
			# next word prediction loss
			outputs = model(**batch)
			loss = outputs.loss
			print(loss)
			loss = loss / GRADIENT_ACCUMULATION_STEPS
			# We keep track of the loss at each epoch
			# if args.with_tracking:
			total_loss += loss.detach().float()
			accelerator.backward(loss)
			if step % GRADIENT_ACCUMULATION_STEPS == 0:
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()

			overall_step += 1

			if isinstance(checkpointing_steps, int):
				output_dir = f"step_{overall_step}"
				if overall_step % checkpointing_steps == 0:
					# if args.output_dir is not None:
					# 	output_dir = os.path.join(args.output_dir, output_dir)
					output_dir = os.path.join(OUTPUT_DIR, output_dir)
					accelerator.save_state(output_dir)

		model.eval()
		for step, batch in enumerate(eval_dataloader):
			# We could avoid this line since we set the accelerator with `device_placement=True`.
			batch.to(accelerator.device)
			with torch.no_grad():
				outputs = model(**batch)

			loss = outputs.loss()
			# ## for now jsut take loss
			# predictions = outputs.logits.argmax(dim=-1)
			# predictions, references = accelerator.gather_for_metrics(
			# 	(predictions, batch["labels"])
			# )
			# metric.add_batch(
			# 	predictions=predictions,
			# 	references=references,
			# )

		# eval_metric = metric.compute()
		eval_metric = loss.item()
		# Use accelerator.print to print only on the main process.
		accelerator.print(f"epoch {epoch}:", eval_metric)
		# if args.with_tracking:
		accelerator.log(
			{
				"eval_loss": eval_metric,
				# "accuracy": eval_metric["accuracy"],
				# "f1": eval_metric["f1"],
				"train_loss": total_loss.item() / len(train_dataloader),
				"epoch": epoch,
			},
			step=epoch,
		)

		if checkpointing_steps == "epoch":
			output_dir = f"epoch_{epoch}"
			# if args.output_dir is not None:
			# output_dir = os.path.join(args.output_dir, output_dir)
			output_dir = os.path.join(OUTPUT_DIR, output_dir)
			accelerator.save_state(output_dir)

	accelerator.end_training()


def main():
	training_function()


if __name__ == "__main__":
	main()
