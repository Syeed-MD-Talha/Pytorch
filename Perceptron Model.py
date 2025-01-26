import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets

# Generate the make_moons dataset
no_of_points = 200
x, y = datasets.make_moons(n_samples=no_of_points, noise=0.2, random_state=123)
xdata = torch.Tensor(x)  # Convert to PyTorch tensor
ydata = torch.Tensor(y.reshape(no_of_points, 1))  # Reshape y to match dimensions

# Function to scatter plot the data
def scatter_plot():
    plt.scatter(x[y == 0, 0], x[y == 0, 1], label="Class 0", c='blue')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], label="Class 1", c='red')
    plt.legend()

# Define the Perceptron model
class PerceptronModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)  # Linear layer

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))  # Apply sigmoid activation
        return pred

    def predict(self, x):
        pred = torch.sigmoid(self.linear(x))
        if pred >= 0.5:
            return 1
        else:
            return 0

# Set random seed for reproducibility
torch.manual_seed(2)

# Create the model, loss function, and optimizer
model = PerceptronModel(2, 1)  # 2 input features, 1 output
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Print initial parameters
print("Initial Parameters:")
print(list(model.parameters()))

# Training loop
epochs = 1000
losses = []
for i in range(epochs):
    ypred = model.forward(xdata)  # Forward pass
    loss = criterion(ypred, ydata)  # Compute loss
    losses.append(loss.item())  # Store loss

    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters

    if (i + 1) % 100 == 0:
        print(f"Epoch: {i + 1}, Loss: {loss.item()}")

# Function to get model parameters
def get_parameters():
    [A, B] = model.parameters()  # Unpack parameters
    A1, A2 = A.view(2)  # Extract weights
    B1 = B[0]  # Extract bias
    return A1.item(), A2.item(), B1.item()

# Function to plot the decision boundary
def plot_decision_boundary(title):
    plt.title(title)
    A1, A2, B1 = get_parameters()  # Get parameters
    x1 = np.array([-2.0, 3.0])  # X-axis range
    y1 = (-B1 - A1 * x1) / A2  # Decision boundary equation
    plt.plot(x1, y1, 'r', label="Decision Boundary")  # Plot boundary
    scatter_plot()  # Plot data points
    plt.legend()
    plt.show()

# Test the model on two new points
p1 = torch.Tensor([1.0, -0.5])  # Point 1
p2 = torch.Tensor([-1.0, 1.5])  # Point 2

# Plot the points
plt.scatter(p1.numpy()[0], p1.numpy()[1], c='green', marker='o', s=100, label="Point 1")
plt.scatter(p2.numpy()[0], p2.numpy()[1], c='purple', marker='x', s=100, label="Point 2")
scatter_plot()
plt.legend()
plt.show()

# Print predictions for the new points
print("Green point positive probability =", model.forward(p1).item())
print("Purple point positive probability =", model.forward(p2).item())
print("Green point in class =", model.predict(p1))
print("Purple point in class =", model.predict(p2))

# Plot the final decision boundary
plot_decision_boundary('Trained Model')
