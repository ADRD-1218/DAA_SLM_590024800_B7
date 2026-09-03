import torch 

x = torch.tensor([1, 2, 3, 4, 5])
print("Tensor:", x)
print("Tensor shape:", x.shape)
print("Tensor dtype:", x.dtype)


print("x + 10:", x + 10)
print("x * 2:", x * 2)
print("x squared:", x ** 2)


A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

B = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

print("A @ B:")
print(A @ B)


print("CUDA available:", torch.cuda.is_available())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


A = A.to(device)
B = B.to(device)

C = A @ B

print("Result:")
print(C)

print("Result device:", C.device)