import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Step 1: Generate dataset
no_of_points = 500
x, y = datasets.make_circles(n_samples=no_of_points, random_state=123, noise=0.1, factor=0.2)

# Print dataset details
unique = np.unique(y)
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)
print("Unique labels:", unique)

# Step 2: Convert data to PyTorch tensors
xdata = torch.tensor(x, dtype=torch.float32)
ydata = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y to match model output

# Step 3: Define the Deep Neural Network model
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.linear2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))  # Apply sigmoid activation to hidden layer
        x = torch.sigmoid(self.linear2(x))  # Apply sigmoid activation to output layer
        return x

    def predict(self, x):
        x = self.forward(x)  # Use the forward method to make predictions
        return (x >= 0.5).int()  # Convert probabilities to binary predictions (0 or 1)

# Step 4: Initialize the model, loss function, and optimizer
input_size = 2  # Number of input features (x and y coordinates)
hidden_size = 4  # Number of neurons in the hidden layer
output_size = 1  # Binary classification (output is 0 or 1)

torch.manual_seed(2)  # Set seed for reproducibility
model = DeepNeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Step 5: Train the model
epochs = 1000
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(xdata)  # Predictions
    loss = criterion(y_pred, ydata)  # Compute loss

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    # Store loss for visualization
    losses.append(loss.item())

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 6: Plot the training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Step 7: Make predictions on new data
new_data = np.array([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
new_data = torch.tensor(new_data, dtype=torch.float32)

# Use the model to predict
predictions = model.predict(new_data)
print("Predictions for new data:")
print(predictions)

# Step 8: Visualize the decision boundary
def plot_decision_boundary(model, x, y):
    # Create a grid of points to evaluate the model
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    # Make predictions on the grid
    with torch.no_grad():
        preds = model.predict(grid_tensor)
        preds = preds.numpy().reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, preds, alpha=0.5, cmap="RdBu")
    plt.scatter(x[y == 0, 0], x[y == 0, 1], label="Class 0", color="red")
    plt.scatter(x[y == 1, 0], x[y == 1, 1], label="Class 1", color="blue")
    plt.legend()
    plt.title("Decision Boundary")
    plt.show()

# Plot the decision boundary
plot_decision_boundary(model, x, y)
