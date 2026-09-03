import torch
import math

torch.manual_seed(42)

X = torch.randn(4, 8)

W_Q = torch.randn(8, 8)
W_K = torch.randn(8, 8)
W_V = torch.randn(8, 8)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

d_k = Q.shape[-1]

scores = (Q @ K.T) / math.sqrt(d_k)

attention_weights = torch.softmax(scores, dim=-1)

output = attention_weights @ V

print("Input shape:", X.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

print("\nAttention weights:")
print(attention_weights)

print("\nRow sums:")
print(attention_weights.sum(dim=-1))

print("\nOutput shape:")
print(output.shape)

seq_len = X.shape[0]

mask = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1
).bool()



print("Mask:")
print(mask)

scores = scores.masked_fill(mask, float("-inf"))

attention_weights = torch.softmax(scores, dim=-1)