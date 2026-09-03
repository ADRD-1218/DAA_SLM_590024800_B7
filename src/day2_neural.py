import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

# Neural Network Dataset
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
]).to(device)

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0]
]).to(device)

model = nn.Linear(1, 1).to(device)

loss_function = nn.MSELoss() #Loss function

#Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):

    # Forward pass
    predictions = model(x)

    # Calculate error
    loss = loss_function(predictions, y)

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss.item():.6f}"
        )

#Parameters of the model
print("\nLearned parameters:")

for name, parameter in model.named_parameters():
    print(name, parameter.data)

test = torch.tensor([[10.0]]).to(device)

prediction = model(test)

#Testing
print("Prediction for x=10:", prediction.item())
