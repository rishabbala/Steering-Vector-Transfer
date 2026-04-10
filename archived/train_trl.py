import torch
from datasets import load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

# load dataset from the HuggingFace Hub
train_dataset = load_dataset("openai/gsm8k", "main", split="train").select(range(100))
eval_dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(100))


def promptFormatter(example, input_name="question", output_name="answer"):
	return {
		"prompt": f"""#### Question: {example[input_name]}\nLet's think step by step.""",
		"completion": f"""#### Answer: {example[output_name]}""",
	}


train_dataset = train_dataset.map(
	promptFormatter,
	fn_kwargs={"input_name": "question", "output_name": "answer"},
	remove_columns=["question", "answer"],
)
eval_dataset = eval_dataset.map(
	promptFormatter,
	fn_kwargs={"input_name": "question", "output_name": "answer"},
	remove_columns=["question", "answer"],
)


model = AutoModelForCausalLM.from_pretrained(
	"meta-llama/Llama-3.1-8B",
	attn_implementation="flash_attention_2",
	torch_dtype=torch.bfloat16,
	device_map="balanced",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id

sft_config = SFTConfig(
	# do_train = True, # not required if we call trainer.train()
	eval_strategy="steps",
	eval_steps=0.5,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	gradient_checkpointing=True,
	gradient_accumulation_steps=2,
	learning_rate=1e-5,
	weight_decay=0.01,
	max_steps=100,
	lr_scheduler_type="linear",
	warmup_ratio=0.1,
	max_length=1024,
	# eos_token=tokenizer.eos_token_id, ## HF code bug return [self._convert_token_to_id_with_added_voc(token) for token in tokens] TypeError: 'int' object is not iterable
	# pad_token=tokenizer.eos_token_id,
	logging_dir="./logs",
	logging_strategy="steps",
	logging_steps=10,
	logging_first_step=True,
	save_strategy="best",
	metric_for_best_model="eval_loss",
	output_dir="/projects/llms-lab/train_partial/llama-3.1-8b-gsm8k/",
	bf16=True,
	# packing=True,
	# padding_free ? explore
	completion_only_loss=True,
	report_to="none",  # Use this for WandB etc
)


model = model.train()
train_params = "v_proj,o_proj,up_proj,gate_proj,down_proj"
train_params = train_params.split(",")
for name, param in model.named_parameters():
	if name not in train_params:
		param.requires_grad_(False)


trainer = SFTTrainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=eval_dataset,
	args=sft_config,
)

trainer.train()
