from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

question = "What is the time complexity of binary search?"

answer = (
    "The time complexity of binary search is O(log n) "
    "because the search space is divided in half after each comparison."
)

text = (
    "Question: "
    + question
    + "\nAnswer: "
    + answer
)

print("Training example:")
print(text)

encoded = tokenizer(
    text,
    return_tensors="pt"
)

print("\nInput IDs:")
print(encoded["input_ids"])

print("\nAttention Mask:")
print(encoded["attention_mask"])

print("\nNumber of tokens:")
print(encoded["input_ids"].shape[1])
