text = "binary search is an efficient searching algorithm"

tokens = text.split()

print("Text:")
print(text)

print("\nTokens:")
print(tokens)


vocabulary = {
    "binary": 1,
    "search": 2,
    "is": 3,
    "an": 4,
    "efficient": 5,
    "searching": 6,
    "algorithm": 7
}

token_ids = [vocabulary[token] for token in tokens]

print("\nToken IDs:")
print(token_ids)