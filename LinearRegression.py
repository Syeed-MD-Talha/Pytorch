import torch

# Define the model
class LinearRegression(torch.nn.Module):
  def __init__(self):
    super().__init__()#The purpose of super() is to ensure the parent class (torch.nn.Module) is properly initialized.
    self.linear = torch.nn.Linear(1, 1) # Define a linear layer with 1 input and 1 output

  def forward(self, x):
    return self.linear(x)  # Apply the linear transformation


inputs=torch.tensor([[1.0],[2.0],[3.0]])
targets=torch.tensor([[2.0],[4.0],[6.0]])

# Create the model, loss function, and optimizer
model = LinearRegression()
loss_function = torch.nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD optimizer

# Training loop
for epoch in range(100):
  # Forward pass
  predictions = model(inputs)  # Compute predictions
  loss = loss_function(predictions, targets)  # Compute loss

  # Backward pass
  optimizer.zero_grad()  # Clear previous gradients
  loss.backward()  # Compute gradients
  optimizer.step()  # Update parameters


new_var=torch.tensor([3.0])
pred_y=model(new_var)
print('Prediction:', pred_y.item())
