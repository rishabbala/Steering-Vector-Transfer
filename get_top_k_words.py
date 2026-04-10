import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
	model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

data = "gsm8k"
model_name = "Meta-Llama-3-8B"
if data == "gsm8k":
	prompt_type = "general-cot"
elif data == "gpqa":
	prompt_type = "mcq-cot"

steer_path = f"/projects/llms-lab/transfer_compare/hs_svd_arch_test/{model_name}/teacher_{model_name}-Instruct_data_{data}_prompt_{prompt_type}.pth"
steer = torch.load(steer_path, map_location="cpu")
steer = torch.FloatTensor(steer)
print("Steering vector shape:", steer.shape)
lm_head = model.get_output_embeddings()
W_out = lm_head.weight
top_k = 20

os.makedirs("vec2words", exist_ok=True)
output_file = os.path.join("vec2words", f"{model_name}-{data}-top-k-words.txt")

with open(output_file, "w", encoding="utf-8") as f:
	for layer_idx, vec in enumerate(steer):
		vec = vec.to(W_out.device, dtype=W_out.dtype)
		scores = torch.matmul(W_out, vec)
		probs = torch.softmax(scores, dim=0)
		topk_probs, topk_idx = torch.topk(probs, k=top_k)
		words = tokenizer.batch_decode(topk_idx)

		layer_header = f"\n--- Layer {layer_idx} ---\n"
		print(layer_header, end="")
		f.write(layer_header)

		for w, p in zip(words, topk_probs.tolist()):
			w_escaped = repr(w)[1:-1].replace(" ", "␣").replace("\t", "␣")
			line = f"{w_escaped:20s} {p:.6f}\n"
			print(line, end="")
			f.write(line)

print(f"\nResults saved to {output_file}")
