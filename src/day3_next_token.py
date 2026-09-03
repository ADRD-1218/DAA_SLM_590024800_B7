text = "binary search is efficient"

tokens = text.split()

for i in range(len(tokens) - 1):
    input_token = tokens[:i + 1]
    target_token = tokens[i + 1]

    print("Input :", input_token)
    print("Target:", target_token)
    print()