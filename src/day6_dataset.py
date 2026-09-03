'''
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("data/daa_train.txt", "r") as f:
    text = f.read()

print("Original text:")
print(text)

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("\nNumber of tokens:", len(tokens))

print("\nFirst 30 tokens:")
print(tokens[:30])

print("\nFirst 30 token IDs:")
print(token_ids[:30])
'''

from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Binary search has logarithmic time complexity."

encoded = tokenizer(
    text,
    return_tensors="pt"
)

print("Input IDs:")
print(encoded["input_ids"])

print("\nAttention Mask:")
print(encoded["attention_mask"])

print("\nShape of Input IDs:")
print(encoded["input_ids"].shape)