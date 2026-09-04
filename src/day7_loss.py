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
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"]
    )

print("Loss:")
print(outputs.loss)