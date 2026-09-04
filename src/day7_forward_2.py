import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

text = "Binary search is an efficient algorithm."

inputs = tokenizer(
    text,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

next_token_logits = outputs.logits[:, -1, :]

next_token_id = torch.argmax(
    next_token_logits,
    dim=-1
)

print("Predicted next token ID:")
print(next_token_id)

print("\nPredicted next token:")
print(tokenizer.decode(next_token_id))