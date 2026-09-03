import torch
import torch.nn as nn

vocab_size = 8
embedding_dimension = 4

embedding = nn.Embedding(
    vocab_size,
    embedding_dimension
)

token_ids = torch.tensor([1, 2, 3, 4])

vectors = embedding(token_ids)

print("Token IDs:")
print(token_ids)

print("\nEmbeddings:")
print(vectors)

print("\nShape:")
print(vectors.shape)