from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "Binary search",
    "Binary search is efficient.",
    "Binary search is an efficient searching algorithm.",
    "Explain why binary search has logarithmic time complexity."
]

for text in texts:
    ids = tokenizer.encode(text)

    print(
        f"{len(ids):3} tokens | {text}"
    )


tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)



print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))

print("\nDecoded text:")
print(tokenizer.decode(token_ids))

print("\nSpecial tokens:")
print(tokenizer.special_tokens_map)