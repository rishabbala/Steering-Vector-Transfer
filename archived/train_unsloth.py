import unsloth  # noqa: F401
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
	"unsloth/Meta-Llama-3.1-8B",
	max_seq_length=512,
	load_in_4bit=False,
	full_finetuning=True,
)

if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer = get_chat_template(
	tokenizer,
	chat_template="llama3",
)


def promptFormatter(examples, input_name="question", output_name="answer"):
	## With chat templats
	convos = [
		[
			{
				"content": "You are a helpful assistant.",
				"role": "system",
			},
			{
				"content": f"""#### Question: {examples[input_name][i]}\nLet's think step by step.""",
				"role": "user",
			},
			{
				"content": f"""#### Answer: {examples[output_name][i]}""",
				"role": "assistant",
			},
		]
		for i in range(len(examples[input_name]))
	]
	texts = [
		tokenizer.apply_chat_template(
			convo,
			tokenize=False,
			add_generation_prompt=True,
		)
		for convo in convos
	]
	return {"text": texts}

	# ## Without chat templates, needs some fix
	# texts = []
	# for i in range(len(examples[input_name])):
	# 	text = f"""#### Question: {examples[input_name][i]}\nLet's think step by step.\n\n"""
	# 	text += f"""#### Answer: {examples[output_name][i]}"""
	# 	texts.append(text)

	# return {"text": texts}


train_dataset = load_dataset("openai/gsm8k", "main", split="train").select(range(100))
eval_dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(100))
# Apply the formatting function to your dataset using the map method
train_dataset = train_dataset.map(
	promptFormatter,
	batched=True,
	fn_kwargs={"input_name": "question", "output_name": "answer"},
)

eval_dataset = eval_dataset.map(
	promptFormatter,
	batched=True,
	fn_kwargs={"input_name": "question", "output_name": "answer"},
)

trainer = SFTTrainer(
	model=model,
	tokenizer=tokenizer,
	train_dataset=train_dataset,
	eval_dataset=eval_dataset,
	args=SFTConfig(
		eval_strategy="steps",
		eval_steps=0.5,
		dataset_text_field="text",
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		gradient_checkpointing=True,
		gradient_accumulation_steps=2,
		warmup_steps=5,
		max_steps=60,
		learning_rate=2e-4,
		weight_decay=0.01,
		optim="adamw_8bit",
		lr_scheduler_type="linear",
		warmup_ratio=0.1,
		max_length=512,
		# eos_token=tokenizer.eos_token_id,
		# pad_token=tokenizer.eos_token_id,
		logging_dir="./logs",
		logging_strategy="steps",
		logging_steps=1,
		logging_first_step=True,
		output_dir="/projects/llms-lab/train_partial/llama-3.1-8b-gsm8k/",
		bf16=True,
		save_strategy="best",
		metric_for_best_model="eval_loss",
		seed=3407,
		report_to="none",  # Use this for WandB etc
		completion_only_loss=True,
	),
)

trainer_stats = trainer.train()
print(trainer_stats)
