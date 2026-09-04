import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

text = "Binary search is an efficient algorithm."

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("Input IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])

print("\nInput Shape:")
print(inputs["input_ids"].shape)

with torch.no_grad():
    outputs = model(**inputs)

print("\nOutput type:")
print(type(outputs))

print("\nLogits shape:")
print(outputs.logits.shape)